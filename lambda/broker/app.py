# lambda/broker/app.py
from __future__ import annotations

import base64
import json
import re
import traceback
from typing import Any, Dict, Optional, Tuple

from companion.broker import ConversationBroker
from companion.config import RuntimeConfig

# Instantiate shared broker components once per container
CONFIG = RuntimeConfig.from_env()
BROKER = ConversationBroker(CONFIG)


# ----- HTTP helpers -----
def _base_headers(extra: Dict[str, str] | None = None) -> Dict[str, str]:
    h = {"content-type": "application/json"}
    if extra:
        h.update({k: v for k, v in extra.items() if v})
    return h


def _ok(body: Dict[str, Any], *, headers: Dict[str, str] | None = None) -> Dict[str, Any]:
    return {
        "statusCode": 200,
        "headers": _base_headers(headers),
        "isBase64Encoded": False,
        "body": json.dumps(body),
    }


def _bad_request(msg: str, *, headers: Dict[str, str] | None = None) -> Dict[str, Any]:
    return {
        "statusCode": 400,
        "headers": _base_headers(headers),
        "isBase64Encoded": False,
        "body": json.dumps({"error": "bad_request", "message": msg}),
    }


def _server_error(exc: Exception, *, headers: Dict[str, str] | None, request_id: str, code: int = 500) -> Dict[str, Any]:
    tb_lines = traceback.format_exc().strip().splitlines()
    tail = tb_lines[-8:] if tb_lines else []
    h = _base_headers(
        {
            "x-lambda-request-id": request_id,
            "x-error-class": exc.__class__.__name__,
            "x-error-msg": (str(exc)[:200] if exc else ""),
            **(headers or {}),
        }
    )
    body = {"error": "internal_error", "message": str(exc), "trace_tail": tail}
    return {"statusCode": code, "headers": h, "isBase64Encoded": False, "body": json.dumps(body)}


# ----- Body parsing -----
def _parse_body(event: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    body = event.get("body")
    if body is None:
        return None, "missing body"
    if isinstance(body, str):
        if event.get("isBase64Encoded"):
            try:
                body = base64.b64decode(body).decode("utf-8", "replace")
            except Exception:
                return None, "body base64 decode failed"
        try:
            body = json.loads(body)
        except Exception:
            return None, "body is not valid JSON"
    if not isinstance(body, dict):
        return None, "body is not an object"
    return body, None


# ----- Flags -----
def _is_truthy(val: Any) -> bool:
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.strip().lower() in {"1", "true", "yes", "y", "on"}
    return False


# ----- Lambda handler -----
def handler(event, context):
    hdr_in = event.get("headers") or {}
    client_session = hdr_in.get("x-client-session") or hdr_in.get("X-Client-Session") or ""
    out_hdr = {"x-client-session": client_session} if client_session else {}

    body, berr = _parse_body(event)
    if berr:
        return _bad_request(berr, headers=out_hdr)

    session_id = (body.get("session_id") or "").strip()
    text = (body.get("text") or "").strip()
    if not session_id:
        return _bad_request("missing or invalid 'session_id'", headers=out_hdr)
    if not text:
        return _bad_request("missing or invalid 'text'", headers=out_hdr)

    context_in = body.get("context") or {}
    voice_id = body.get("voice_id")
    if isinstance(voice_id, str):
        voice_id = voice_id.strip() or None
    else:
        voice_id = None
    text_only = _is_truthy(body.get("text_only"))

    # Identity fast-path
    speaker = context_in.get("speaker_id") if isinstance(context_in, dict) else None
    if speaker and re.search(r"\b(who am i|what'?s my name|who'?s speaking)\b", text.lower()):
        reply_text = f"You are {speaker}."
        payload = {"text": reply_text, "command": {"name": "noop", "args": {}}, "audio": {"audio_base64": None}}
        if text_only:
            return _ok(payload, headers=out_hdr)
        try:
            audio = BROKER._speech.synthesize(  # type: ignore[attr-defined]
                text=reply_text,
                session_id=session_id,
                response_id=str(getattr(context, "aws_request_id", "0")),
                voice_id=voice_id,
            )
            payload["audio"] = audio
        except Exception:
            pass
        return _ok(payload, headers=out_hdr)

    try:
        broker_resp = BROKER.handle(
            session_id=session_id,
            user_text=text,
            context=context_in if isinstance(context_in, dict) else {},
            text_only=text_only,
            voice_id=voice_id,
        )
    except Exception as exc:
        return _server_error(exc, headers=out_hdr, request_id=getattr(context, "aws_request_id", "unknown"), code=502)

    audio_payload = broker_resp.get("audio") or {"audio_base64": None}
    if text_only and isinstance(audio_payload, dict):
        audio_payload["audio_base64"] = None
    return _ok(
        {
            "text": broker_resp.get("text", ""),
            "command": broker_resp.get("command") or {"name": "noop", "args": {}},
            "audio": audio_payload,
        },
        headers=out_hdr,
    )
