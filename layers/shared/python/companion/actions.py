# layers/shared/python/companion/actions.py
"""Server-side action bridge to the outside world.

This module lets the AI companion trigger side-effectful actions from the
Lambda broker: SMS, email, and a small whitelist of system commands.

Everything is OFF by default. Enable explicitly via environment variables:

    ENABLE_OUTBOUND_SMS=1         # allow send_text
    ENABLE_OUTBOUND_EMAIL=1       # allow send_email
    ENABLE_SYSTEM_COMMANDS=1      # allow run_command

You also need a verified SES sender address for email:

    ACTION_EMAIL_FROM=you@example.com
    # or SES_FROM_ADDRESS=you@example.com

These env vars are read inside the Lambda runtime, not the client.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from typing import Any, Dict, Mapping

import boto3


_BOOL_TRUE = {"1", "true", "yes", "y", "on"}


def _flag(name: str, default: bool = False) -> bool:
    """Parse a boolean-ish env var."""
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in _BOOL_TRUE


# Very conservative whitelist of system commands to run in Lambda.
DEFAULT_ALLOWED_COMMANDS: Mapping[str, list[str]] = {
    "uptime": ["uptime"],
    "disk_usage": ["df", "-h"],
    "who": ["who"],
}


@dataclass(slots=True)
class ActionResult:
    """Lightweight result for logging + optional user-visible status."""

    name: str
    executed: bool
    error: str | None = None
    detail: str | None = None
    visible_message: str | None = None


class ActionManager:
    """Executes high-impact actions with simple config-based gating.

    This does NOT try to implement a full approval protocol. It just enforces
    coarse-grained enable/disable flags that you control via env vars.
    """

    def __init__(self, region_name: str) -> None:
        self._region = region_name

        # Feature flags (all disabled by default)
        self._enable_sms = _flag("ENABLE_OUTBOUND_SMS", False)
        self._enable_email = _flag("ENABLE_OUTBOUND_EMAIL", False)
        self._enable_cmd = _flag("ENABLE_SYSTEM_COMMANDS", False)

        # Lazily-initialised AWS clients
        self._sns = boto3.client("sns", region_name=region_name) if self._enable_sms else None
        self._ses = boto3.client("ses", region_name=region_name) if self._enable_email else None

        # For SES: verified sender address
        self._email_from = os.getenv("ACTION_EMAIL_FROM", "").strip() or os.getenv(
            "SES_FROM_ADDRESS", ""
        ).strip()

    # ---- Public API -----------------------------------------------------

    def run_server_action(self, name: str, args: Dict[str, Any]) -> ActionResult:
        """Execute an action synchronously in the Lambda environment.

        Supported actions:
        - send_text:  {"to": "+12065551234", "body": "..."}
        - send_email: {"to": "user@example.com", "subject": "...", "body": "..."}
        - run_command: {"name": "uptime" | "disk_usage" | "who"}
        """
        try:
            if name == "send_text":
                return self._send_text(args)
            if name == "send_email":
                return self._send_email(args)
            if name == "run_command":
                return self._run_command(args)
        except Exception as exc:  # pragma: no cover - best-effort safety
            return ActionResult(
                name=name,
                executed=False,
                error=f"{type(exc).__name__}: {exc}",
            )

        return ActionResult(
            name=name,
            executed=False,
            error="Unknown action name.",
        )

    # ---- Individual actions ---------------------------------------------

    def _send_text(self, args: Dict[str, Any]) -> ActionResult:
        if not self._enable_sms or self._sns is None:
            return ActionResult(
                name="send_text",
                executed=False,
                error="Outbound SMS disabled (set ENABLE_OUTBOUND_SMS=1 to enable).",
            )

        to = str(args.get("to") or "").strip()
        body = str(args.get("body") or "").strip()

        if not to or not body:
            return ActionResult(
                name="send_text",
                executed=False,
                error="send_text requires 'to' (E.164 phone) and 'body'.",
            )

        resp = self._sns.publish(PhoneNumber=to, Message=body)
        msg_id = resp.get("MessageId") or "unknown"

        return ActionResult(
            name="send_text",
            executed=True,
            detail=f"message_id={msg_id}",
            visible_message=f"[system] Sent text message to {to}.",
        )

    def _send_email(self, args: Dict[str, Any]) -> ActionResult:
        if not self._enable_email or self._ses is None:
            return ActionResult(
                name="send_email",
                executed=False,
                error="Outbound email disabled (set ENABLE_OUTBOUND_EMAIL=1 to enable).",
            )

        to = str(args.get("to") or "").strip()
        subject = str(args.get("subject") or "").strip() or "(no subject)"
        body = str(args.get("body") or "").strip()

        if not to:
            return ActionResult(
                name="send_email",
                executed=False,
                error="send_email requires 'to'.",
            )

        if not self._email_from:
            return ActionResult(
                name="send_email",
                executed=False,
                error="ACTION_EMAIL_FROM or SES_FROM_ADDRESS must be set for send_email.",
            )

        resp = self._ses.send_email(
            Source=self._email_from,
            Destination={"ToAddresses": [to]},
            Message={
                "Subject": {"Data": subject},
                "Body": {"Text": {"Data": body}},
            },
        )
        msg_id = resp.get("MessageId") or "unknown"

        return ActionResult(
            name="send_email",
            executed=True,
            detail=f"message_id={msg_id}",
            visible_message=f"[system] Sent email to {to}.",
        )

    def _run_command(self, args: Dict[str, Any]) -> ActionResult:
        if not self._enable_cmd:
            return ActionResult(
                name="run_command",
                executed=False,
                error="System commands disabled (set ENABLE_SYSTEM_COMMANDS=1 to enable).",
            )

        name = str(args.get("name") or "").strip()
        if not name:
            return ActionResult(
                name="run_command",
                executed=False,
                error="run_command requires 'name'.",
            )

        cmd = DEFAULT_ALLOWED_COMMANDS.get(name)
        if not cmd:
            return ActionResult(
                name="run_command",
                executed=False,
                error=f"Command {name!r} is not in the allowed list.",
            )

        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
        )
        output = (completed.stdout or completed.stderr or "").strip()
        detail = f"exit_code={completed.returncode}"
        if output:
            detail += f", output={output!r}"

        return ActionResult(
            name="run_command",
            executed=True,
            detail=detail,
            visible_message=f"[system] Ran {name!r}: exit={completed.returncode}.",
        )


__all__ = ["ActionManager", "ActionResult", "DEFAULT_ALLOWED_COMMANDS"]
