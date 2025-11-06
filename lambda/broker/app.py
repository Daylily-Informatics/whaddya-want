"""Lambda entry point for the conversational broker."""
from __future__ import annotations

import base64
import json
import sys
import traceback
import uuid
from typing import Any

from companion import ConversationBroker, RuntimeConfig

_CONFIG = RuntimeConfig.from_env()
_BROKER = ConversationBroker(_CONFIG)


def _resp(status: int, payload: dict[str, Any], *, is_b64: bool = False, headers: dict[str, str] | None = None) -> dict[str, Any]:
    return {
        "statusCode": status,
        "headers": {"content-type": "application/json", **(headers or {})},
        "isBase64Encoded": is_b64,
        "body": json.dumps(payload),
    }


def handler(event: dict[str, Any], context: Any) -> dict[str, Any]:  # pragma: no cover - exercised in integration
    try:
        print("EVENT:", json.dumps(event)[:2000], file=sys.stdout)
    except Exception:
        pass

    try:
        body = event.get("body") or "{}"
        if event.get("isBase64Encoded"):
            body = base64.b64decode(body).decode("utf-8", "ignore")
        data = json.loads(body)

        text_in = (data.get("text") or "").strip()
        session_id = (data.get("session_id") or "").strip() or str(uuid.uuid4())
        if not text_in:
            return _resp(400, {"error": "missing 'text'"})

        audio_metadata: dict[str, Any] = data.get("audio") or {}
        context = {
            "speaker": data.get("speaker")
            or audio_metadata.get("speaker")
            or audio_metadata.get("speaker_label"),
            "sound_type": data.get("sound_type") or audio_metadata.get("sound_type"),
        }
        context = {k: v for k, v in context.items() if v}

        result = _BROKER.handle(session_id=session_id, user_text=text_in, context=context or None)

        payload: dict[str, Any] = {
            "text": result["text"],
            "audio": result["audio"],
        }
        if result.get("tool_calls"):
            payload["tool_calls"] = result["tool_calls"]
        if result.get("classification"):
            payload["classification"] = result["classification"]

        return _resp(200, payload)

    except Exception as exc:  # pragma: no cover - defensive logging
        print("ERROR:", exc, file=sys.stderr)
        traceback.print_exc()
        return _resp(500, {"error": "broker error"})
