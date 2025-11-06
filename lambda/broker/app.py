# lambda/broker/app.py
from __future__ import annotations

import os, json, base64, re, time, traceback
from typing import List, Dict, Any, Optional, Tuple

import boto3
from botocore.config import Config

# ----- Env -----
REGION: str = os.environ.get("AWS_REGION", "us-west-2")
POLLY_VOICE: str = os.environ.get("POLLY_VOICE", "Joanna")
MODEL_ID: str = os.environ.get("MODEL_ID", "amazon.titan-text-express-v1")

HISTORY_LIMIT: int = int(os.environ.get("HISTORY_LIMIT", "12"))
USE_MEMORY: bool = os.environ.get("USE_MEMORY", "false").lower() == "true"
TABLE: Optional[str] = os.environ.get("CONVERSATION_TABLE")  # only used if USE_MEMORY=true

# ----- Clients -----
_cfg = Config(retries={"max_attempts": 3, "mode": "standard"})
polly = boto3.client("polly", region_name=REGION, config=_cfg)
brt   = boto3.client("bedrock-runtime", region_name=REGION, config=_cfg)

dynamodb = boto3.resource("dynamodb", region_name=REGION) if USE_MEMORY and TABLE else None
mem_table = dynamodb.Table(TABLE) if dynamodb else None


# ----- HTTP helpers -----
def _base_headers(extra: Dict[str, str] | None = None) -> Dict[str, str]:
    h = {"content-type": "application/json"}
    if extra:
        h.update({k: v for k, v in extra.items() if v})
    return h

def _ok(body: Dict[str, Any], *, headers: Dict[str, str] | None = None) -> Dict[str, Any]:
    return {"statusCode": 200, "headers": _base_headers(headers), "isBase64Encoded": False, "body": json.dumps(body)}

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


# ----- Memory -----
def _get_memory(session_id: str) -> List[Dict[str, str]]:
    if not mem_table:
        return []
    item = mem_table.get_item(Key={"session_id": session_id}).get("Item")
    return item.get("turns", []) if item else []

def _put_memory(session_id: str, turns: List[Dict[str, str]]) -> None:
    if not mem_table:
        return
    turns = turns[-HISTORY_LIMIT:]
    mem_table.put_item(Item={"session_id": session_id, "turns": turns, "ttl": int(time.time()) + 7 * 24 * 3600})


# ----- Prompting -----
def _system_prompt(speaker: Optional[str], acoustic_event: Optional[str]) -> str:
    lines = [
        "You are Forge, a concise, capable home/office companion.",
        "Be natural, brief, and helpful. Avoid fluff.",
        "If asked who the user is, answer directly.",
    ]
    if speaker:
        lines.append(f"Current speaker: {speaker}. Use their name naturally.")
    if acoustic_event == "dog_bark":
        lines.append("A dog bark was detected recently; briefly acknowledge then continue.")
    return "\n".join(lines)


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


# ----- TTS -----
def _tts(text: str, voice_id: Optional[str] = None) -> Dict[str, Any]:
    """Synthesize speech. Raises on Polly errors (caught in handler)."""
    voice = voice_id or POLLY_VOICE
    r = polly.synthesize_speech(Text=text, VoiceId=voice, OutputFormat="mp3")
    audio = r["AudioStream"].read()
    return {"text": text, "audio": {"audio_base64": base64.b64encode(audio).decode()}}


# ----- LLM calls -----
def _call_titan_text_express(system: str, history: List[Dict[str, str]], user_text: str) -> str:
    # Titan doesn't have a separate system channel. Embed it into plain text.
    parts = []
    if system:
        parts.append(f"System:\n{system}\n")
    for t in history:
        if t.get("user"):
            parts.append(f"User:\n{t['user']}\n")
        if t.get("assistant"):
            parts.append(f"Assistant:\n{t['assistant']}\n")
    user_text = user_text or "Respond briefly."
    parts.append(f"User:\n{user_text}\nAssistant:\n")
    prompt = "\n".join(parts)

    payload = {
        "inputText": prompt,
        "textGenerationConfig": {
            "maxTokenCount": 300,
            "temperature": 0.4,
            "topP": 0.9,
            "stopSequences": ["\nUser:", "\nSystem:"],
        },
    }

    resp = brt.invoke_model(
        modelId="amazon.titan-text-express-v1",  # lock to Titan schema here
        contentType="application/json",
        accept="application/json",
        body=json.dumps(payload),
    )
    out = json.loads(resp["body"].read())
    results = out.get("results") or []
    if not results:
        return "I couldn’t generate a response."
    return results[0].get("outputText") or "I couldn’t generate a response."


def _call_converse(model_id: str, system: str, history: List[Dict[str, str]], user_text: str) -> str:
    """Model-agnostic Bedrock Converse call; retry without system if unsupported."""
    def _msgs():
        msgs: List[Dict[str, Any]] = []
        for t in history:
            if "user" in t:
                msgs.append({"role": "user", "content": [{"text": t["user"]}]})
            if "assistant" in t:
                msgs.append({"role": "assistant", "content": [{"text": t["assistant"]}]})
        msgs.append({"role": "user", "content": [{"text": user_text or "Respond briefly."}]})
        return msgs

    kwargs: Dict[str, Any] = {
        "modelId": model_id,
        "messages": _msgs(),
        "inferenceConfig": {"maxTokens": 300, "temperature": 0.4, "topP": 0.9},
    }
    had_system = False
    if system:
        kwargs["system"] = [{"text": system}]
        had_system = True

    try:
        out = brt.converse(**kwargs)
        return out["output"]["message"]["content"][0]["text"]
    except Exception as e:
        msg = str(e)
        if had_system and ("system" in msg.lower() or "doesn't support system" in msg.lower()):
            kwargs.pop("system", None)
            out = brt.converse(**kwargs)
            return out["output"]["message"]["content"][0]["text"]
        raise

def _call_llm(model_id: str, system: str, history: List[Dict[str, str]], user_text: str) -> str:
    if model_id.startswith("amazon.titan-text-express"):
        return _call_titan_text_express(system, history, user_text)
    return _call_converse(model_id, system, history, user_text)


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
    text       = (body.get("text") or "").strip()
    if not session_id:
        return _bad_request("missing or invalid 'session_id'", headers=out_hdr)
    if not text:
        return _bad_request("missing or invalid 'text'", headers=out_hdr)

    context_in = body.get("context") or {}
    speaker    = context_in.get("speaker_id")
    acoustic   = context_in.get("acoustic_event")

    voice_id   = body.get("voice_id")
    if isinstance(voice_id, str):
        voice_id = voice_id.strip() or None
    else:
        voice_id = None

    # Identity fast-path
    if speaker and re.search(r"\b(who am i|what'?s my name|who'?s speaking)\b", text.lower()):
        try:
            return _ok(_tts(f"You are {speaker}.", voice_id), headers=out_hdr)
        except Exception as exc:
            # If Polly fails, still give a 200 with text only to avoid breaking UX on identity
            return _ok({"text": f"You are {speaker}.", "audio": {"audio_base64": None}}, headers=out_hdr)

    # Short memory (optional)
    try:
        history: List[Dict[str, str]] = _get_memory(session_id) if USE_MEMORY else []
    except Exception as exc:
        # Memory failures should not 500 the whole request
        history = []

    system = _system_prompt(speaker, acoustic)

    # Generate reply (LLM)
    try:
        reply = _call_llm(MODEL_ID, system, history, text)
    except Exception as exc:
        # Return structured 502 so the client can see cause immediately
        return _server_error(exc, headers=out_hdr, request_id=getattr(context, "aws_request_id", "unknown"), code=502)

    # Update memory (best-effort)
    try:
        if USE_MEMORY:
            history.append({"user": text, "assistant": reply})
            _put_memory(session_id, history)
    except Exception:
        pass

    # TTS (Polly) — if Polly fails, report a 502 with details instead of 500
    try:
        return _ok(_tts(reply, voice_id), headers=out_hdr)
    except Exception as exc:
        return _server_error(exc, headers=out_hdr, request_id=getattr(context, "aws_request_id", "unknown"), code=502)
