"""AWS Lambda handler for the AI companion broker."""
from __future__ import annotations

import base64
import json
from http import HTTPStatus
from typing import Any

from companion import ConversationBroker, RuntimeConfig


CONFIG = RuntimeConfig.from_env()
BROKER = ConversationBroker(CONFIG)


def _parse_body(event: dict[str, Any]) -> dict[str, Any]:
    if "body" not in event:
        raise ValueError("Event missing body")
    raw_body = event["body"]
    if event.get("isBase64Encoded"):
        decoded = base64.b64decode(raw_body)
        body = json.loads(decoded)
    else:
        body = json.loads(raw_body)
    return body


def handler(event: dict[str, Any], _: Any) -> dict[str, Any]:
    try:
        payload = _parse_body(event)
        session_id = payload["session_id"]
        user_text = payload["text"]
        context = payload.get("context")
        result = BROKER.handle(session_id=session_id, user_text=user_text, context=context)
        return {
            "statusCode": HTTPStatus.OK,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps(result),
        }
    except Exception as exc:  # pragma: no cover - network/service failure surfaces here
        return {
            "statusCode": HTTPStatus.INTERNAL_SERVER_ERROR,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"error": str(exc)}),
        }
