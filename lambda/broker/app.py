# lambda/broker/app.py
from __future__ import annotations

import base64
import json
import re
import time
import traceback
from typing import Any, Dict, Optional, Tuple

from companion import ConversationBroker, RuntimeConfig
from companion.vision import VisionClient, VisionConfig


# ----- Broker singleton -----
_CONFIG = RuntimeConfig.from_env()
_BROKER = ConversationBroker(_CONFIG)

_VISION: Optional[VisionClient] = None
if _CONFIG.vision_model_id:
    _VISION = VisionClient(
        VisionConfig(region_name=_CONFIG.region_name, model_id=_CONFIG.vision_model_id)
    )

# ----- Short-lived vision cache (per process, per session) -----
# This lets us reuse the latest vision_scene for follow-up questions like
# "what color are the curtains?" without forcing every request to carry an image.
_LAST_VISION_BY_SESSION: Dict[str, Dict[str, Any]] = {}
VISION_MAX_AGE_SEC = 12.0


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


def _server_error(
    exc: Exception,
    *,
    headers: Dict[str, str] | None,
    request_id: str,
    code: int = 500,
) -> Dict[str, Any]:
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
    # Correlate with client
    hdr_in = event.get("headers") or {}
    client_session = hdr_in.get("x-client-session") or hdr_in.get("X-Client-Session") or ""
    out_hdr = {"x-client-session": client_session} if client_session else {}

    # Parse body
    body, berr = _parse_body(event)
    if berr:
        return _bad_request(berr, headers=out_hdr)

    # Validate inputs early → clear 4xx instead of mystery 500s
    session_id = (body.get("session_id") or "").strip()
    text = (body.get("text") or "").strip()
    if not session_id:
        return _bad_request("missing or invalid 'session_id'", headers=out_hdr)
    if not text:
        return _bad_request("missing or invalid 'text'", headers=out_hdr)

    context_raw = body.get("context")
    if isinstance(context_raw, dict):
        context_in: Dict[str, Any] = dict(context_raw)
    else:
        context_in = {}

    speaker = context_in.get("speaker_id")
    text_only = _is_truthy(body.get("text_only"))

    voice_id = body.get("voice_id")
    if isinstance(voice_id, str):
        voice_id = voice_id.strip() or None
    else:
        voice_id = None

    # Optional: run a vision pass if an inline image was provided
    image_b64 = body.get("image_base64")
    vision_scene: Optional[Dict[str, Any]] = None
    if _VISION is not None and isinstance(image_b64, str):
        payload = image_b64.strip()
        if payload:
            try:
                img_bytes = base64.b64decode(payload, validate=True)
            except Exception:
                img_bytes = None
            if img_bytes:
                try:
                    hint_val = context_in.get("vision_hint")
                    hint = hint_val if isinstance(hint_val, str) and hint_val.strip() else None
                    vision_scene = _VISION.describe_scene(img_bytes, hint=hint)
                except Exception as exc:
                    # Don't kill the whole request if vision fails; just surface the error.
                    vision_scene = {"error": str(exc)}

    # Attach fresh scene for this call and update cache
    if vision_scene is not None:
        context_in["vision_scene"] = vision_scene
        _LAST_VISION_BY_SESSION[session_id] = {
            "scene": vision_scene,
            "ts": time.time(),
        }
    else:
        # No inline image this time; if we have a recent cached scene for this
        # session, attach it so follow-up environment questions can see it.
        cached = _LAST_VISION_BY_SESSION.get(session_id)
        if cached and "vision_scene" not in context_in:
            try:
                ts = float(cached.get("ts", 0.0))
            except (TypeError, ValueError):
                ts = 0.0
            age = time.time() - ts
            if age <= VISION_MAX_AGE_SEC:
                context_in["vision_scene"] = cached["scene"]

    # Identity fast-path: cheap and deterministic; text-only on purpose.
    if speaker and re.search(r"\b(who am i|what'?s my name|who'?s speaking)\b", text.lower()):
        plain = {
            "text": f"You are {speaker}.",
            "audio": {"audio_base64": None},
            "command": {"name": "noop", "args": {}},
        }
        return _ok(plain, headers=out_hdr)

    # Generate reply via canonical ConversationBroker
    try:
        broker_out = _BROKER.handle(
            session_id=session_id,
            user_text=text,
            context=context_in,
            voice_id=voice_id,
            text_only=text_only,
        )
    except Exception as exc:
        # Return structured 502 so the client can see cause immediately
        return _server_error(
            exc,
            headers=out_hdr,
            request_id=getattr(context, "aws_request_id", "unknown"),
            code=502,
        )

    reply = str(broker_out.get("text") or "")
    audio_payload = broker_out.get("audio") or {}
    command = broker_out.get("command") or {"name": "noop", "args": {}}

    body_out: Dict[str, Any] = {
        "text": reply,
        "audio": audio_payload,
        "command": command,
    }
    return _ok(body_out, headers=out_hdr)
