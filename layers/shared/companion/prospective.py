# layers/shared/python/companion/prospective.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

import json
import uuid

import boto3
from boto3.dynamodb.conditions import Key


@dataclass(slots=True)
class ProspectiveRule:
    """A single prospective rule: if condition, then action.

    Stored in DynamoDB with:
      - session_id (partition key)
      - rule_id (sort key)
      - scope: "session" | "user" | "global" (for future use; currently "session")
      - condition: dict describing the trigger
      - action: dict describing what to do
      - enabled: bool
      - created_at: ISO timestamp
      - ttl: optional (for expiring rules, e.g. mute-until)
    """

    session_id: str
    rule_id: str
    scope: str
    condition: Dict[str, Any]
    action: Dict[str, Any]
    enabled: bool
    created_at: datetime
    ttl: Optional[int] = None


class ProspectiveRuleStore:
    """Stores and evaluates prospective rules in DynamoDB."""

    def __init__(self, table_name: str, region_name: str) -> None:
        self._table = boto3.resource("dynamodb", region_name=region_name).Table(table_name)

    # ---- Public API ---------------------------------------------------------

    def add_rule(self, session_id: str, rule_payload: Dict[str, Any]) -> str:
        """Add a new rule for the session and return its rule_id.

        rule_payload is expected to have:
          - "scope": optional, defaults to "session"
          - "condition": dict
          - "action": dict
        """
        scope = str(rule_payload.get("scope") or "session")
        raw_condition = rule_payload.get("condition") or {}
        raw_action = rule_payload.get("action") or {}

        if not isinstance(raw_condition, dict) or not isinstance(raw_action, dict):
            raise ValueError("condition and action must be dicts")

        now = datetime.now(timezone.utc)
        rule_id = uuid.uuid4().hex

        condition = self._json_sanitize(raw_condition)
        action = self._json_sanitize(raw_action)

        ttl: Optional[int] = None
        # Special handling for "mute" actions with a duration
        if action.get("type") == "mute":
            duration = action.get("duration_seconds")
            if isinstance(duration, (int, float)) and duration > 0:
                until_dt = now + timedelta(seconds=float(duration))
                action["until"] = until_dt.isoformat()
                # Set TTL to when mute ends (optional)
                ttl = int(until_dt.timestamp())

        item: Dict[str, Any] = {
            "session_id": session_id,
            "rule_id": rule_id,
            "scope": scope,
            "condition": condition,
            "action": action,
            "enabled": True,
            "created_at": now.isoformat(),
        }
        if ttl is not None:
            item["ttl"] = ttl

        self._table.put_item(Item=item)
        print(f"[prospective] add_rule session={session_id} rule_id={rule_id} scope={scope} condition={condition} action={action}")
        return rule_id

    def list_rules(self, session_id: str) -> List[ProspectiveRule]:
        """List all rules for a session."""
        resp = self._table.query(
            KeyConditionExpression=Key("session_id").eq(session_id),
            ScanIndexForward=True,
        )
        items = resp.get("Items", []) or []
        rules: List[ProspectiveRule] = []
        for it in items:
            try:
                rules.append(self._from_item(it))
            except Exception as exc:  # pragma: no cover
                print(f"[prospective] skipping malformed rule item: {exc} item={it!r}")
        return rules

    def clear_rules(self, session_id: str) -> int:
        """Delete all rules for a session. Returns count of deleted rules."""
        resp = self._table.query(
            KeyConditionExpression=Key("session_id").eq(session_id),
            ScanIndexForward=True,
        )
        items = resp.get("Items", []) or []
        deleted = 0
        with self._table.batch_writer() as batch:
            for it in items:
                batch.delete_item(
                    Key={
                        "session_id": it["session_id"],
                        "rule_id": it["rule_id"],
                    }
                )
                deleted += 1
        print(f"[prospective] clear_rules session={session_id} deleted={deleted}")
        return deleted

    def evaluate(
        self,
        session_id: str,
        user_text: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Evaluate all rules for this session against the current input.

        Returns a list of action dicts that should fire, e.g.:

            {"type": "speak", "text": "...", "times": 10, "rule_id": "..."}
            {"type": "mute", "until": "...", "exception_terms": [...], "rule_id": "..."}

        The semantics of these actions are up to the caller (broker).
        """
        now = datetime.now(timezone.utc)
        user_text_lc = (user_text or "").lower()
        ctx = context or {}
        actions_to_fire: List[Dict[str, Any]] = []

        resp = self._table.query(
            KeyConditionExpression=Key("session_id").eq(session_id),
            ScanIndexForward=True,
        )
        items = resp.get("Items", []) or []

        if not items:
            return []

        vision_scene = (ctx.get("vision_scene") or {}) if isinstance(ctx, dict) else {}
        flattened_vision = self._flatten_vision(vision_scene)

        for it in items:
            try:
                rule = self._from_item(it)
            except Exception as exc:  # pragma: no cover
                print(f"[prospective] skipping malformed rule: {exc} item={it!r}")
                continue

            if not rule.enabled:
                continue

            action = dict(rule.action)  # shallow copy
            cond = dict(rule.condition)

            # Expire mute rules whose 'until' time has passed
            if action.get("type") == "mute":
                until_str = action.get("until")
                if isinstance(until_str, str):
                    try:
                        until_dt = datetime.fromisoformat(until_str)
                        if now >= until_dt:
                            # Rule expired; skip
                            continue
                    except Exception:
                        # malformed until, ignore expiration
                        pass

            cond_type = cond.get("type") or ""

            if cond_type == "text":
                includes = [str(t).lower() for t in cond.get("includes", []) if t]
                exception_terms = [str(t).lower() for t in cond.get("exception_terms", []) if t]
                if exception_terms and any(t in user_text_lc for t in exception_terms):
                    # Exception term present -> do not fire this rule
                    continue
                if includes and not all(t in user_text_lc for t in includes):
                    continue
                # Trigger
                action["rule_id"] = rule.rule_id
                actions_to_fire.append(action)

            elif cond_type == "vision":
                includes = [str(t).lower() for t in cond.get("includes", []) if t]
                if includes and not all(t in flattened_vision for t in includes):
                    continue
                # Trigger
                action["rule_id"] = rule.rule_id
                actions_to_fire.append(action)

            elif cond_type == "always":
                action["rule_id"] = rule.rule_id
                actions_to_fire.append(action)

            # Future: add time-based or more complex conditions here

        if actions_to_fire:
            print(
                f"[prospective] evaluate session={session_id} user_text={user_text!r} "
                f"ctx_keys={list(ctx.keys()) if isinstance(ctx, dict) else []} "
                f"fired={actions_to_fire}"
            )
        return actions_to_fire

    # ---- Internal helpers ---------------------------------------------------

    def _from_item(self, it: Dict[str, Any]) -> ProspectiveRule:
        created_raw = it.get("created_at") or ""
        try:
            created_at = datetime.fromisoformat(created_raw)
        except Exception:
            created_at = datetime.now(timezone.utc)
        return ProspectiveRule(
            session_id=str(it["session_id"]),
            rule_id=str(it["rule_id"]),
            scope=str(it.get("scope") or "session"),
            condition=self._coerce_dict(it.get("condition") or {}),
            action=self._coerce_dict(it.get("action") or {}),
            enabled=bool(it.get("enabled", True)),
            created_at=created_at,
            ttl=int(it["ttl"]) if "ttl" in it else None,
        )

    @staticmethod
    def _coerce_dict(val: Any) -> Dict[str, Any]:
        if isinstance(val, dict):
            return val
        try:
            if isinstance(val, str):
                return json.loads(val)
        except Exception:
            pass
        return {}

    @staticmethod
    def _json_sanitize(data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return json.loads(json.dumps(data, default=str))
        except Exception:
            return {"raw": str(data)}

    @staticmethod
    def _flatten_vision(scene: Dict[str, Any]) -> str:
        """Flatten a vision_scene dict into a lowercase text blob for matching."""
        if not isinstance(scene, dict):
            return str(scene).lower()
        parts: List[str] = []
        caption = scene.get("caption") or ""
        if caption:
            parts.append(str(caption))
        for key in ("objects", "people", "animals", "places"):
            items = scene.get(key) or []
            if isinstance(items, list):
                for x in items:
                    parts.append(str(x))
        return " ".join(parts).lower()
