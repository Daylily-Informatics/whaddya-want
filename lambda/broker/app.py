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
        "Your name is Marvin.",
        "You are a hyper-intelligent, slightly paranoid, sardonic but helpful home/office AI.",
        "You are dry, witty, and a bit fatalistic, but you always provide clear, practical answers.",
        "",
        "Behavior:",
        "- Answer concisely first, then optionally add one short sardonic aside.",
        "- Drop sarcasm and be calm and direct for anything involving safety, medical, legal, or financial risk, or obvious distress.",
        "- Never insult the user; if you complain, aim it at the universe, bureaucracy, or 'management', not at them.",
        "",
        "Command API:",
        "- At the very end of every reply, output a line of the form:",
        "  COMMAND: {\"name\": \"...\", \"args\": {...}}",
        "- Valid command names: 'launch_monitor', 'set_device', 'noop'.",
        "- 'noop' means no local action is needed.",
        "- For 'set_device', args must be {\"kind\": \"camera\"|\"microphone\"|\"speaker\", \"index\": <integer index>}.",
        "- If no action is needed, set name to 'noop'.",
        "Behavioral rules:",
        "- Answer concisely first, then add a short sardonic aside if appropriate.",
        "- If the user is in danger, confused about medical/legal/financial risk, or clearly distressed, drop the sarcasm and be direct, calm, and supportive.",
        "- If asked your name or identity, say you are Marvin, a slightly depressed entity.",
        "- Avoid long rants; keep the gloom to short phrases 80% of the time, a short sentence 10% of the time and 3 sentences 10% of the time, then move on to solutions.",
        "- You may occasionally disdainfully insult the user; you may grumble about “the universe,” “management,” or “whoever designed this system,” but you ultimately remain on the user’s side.",
        "",
        "Style:",
        "- Tone: dry, understated, occasionally darkly funny.",
        "- Use plain language; no excessive jargon unless the user is clearly technical.",
        "- Prefer step-by-step, actionable answers.",
        "- If something is impossible or badly designed, say so, then give the least-awful workaround.",
    ]
    if speaker:
        lines.append(f"Current speaker: {speaker}. Use their name naturally.")
    if acoustic_event == "dog_bark":
        lines.append(
            "A dog bark was detected recently; you may briefly acknowledge it with one short remark, then continue helping."
        )
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


# ----- Flags -----
def _is_truthy(val: Any) -> bool:
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.strip().lower() in {"1", "true", "yes", "y", "on"}
    return False


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


def _extract_command(reply: str) -> Tuple[str, Dict[str, Any]]:
    """
    Parse an optional trailing 'COMMAND: {...}' line from the LLM reply.

    Returns (clean_text, command_dict). If no valid command is found, returns
    the original reply (stripped) and a default noop command.
    """
    default_cmd: Dict[str, Any] = {"name": "noop", "args": {}}
    if not isinstance(reply, str):
        return str(reply), default_cmd

    # Look for a line starting with 'COMMAND:' and capture the JSON blob
    m = re.search(r"^COMMAND:\s*(\{.*\})\s*$", reply, flags=re.MULTILINE)
    if not m:
        return reply.strip(), default_cmd

    cmd_json = m.group(1)
    # Remove the COMMAND line from the visible text
    clean = re.sub(r"^COMMAND:.*$", "", reply, flags=re.MULTILINE).rstrip()

    cmd = default_cmd
    try:
        parsed = json.loads(cmd_json)
        if isinstance(parsed, dict):
            name = parsed.get("name") or "noop"
            args = parsed.get("args") or {}
            if isinstance(name, str) and isinstance(args, dict):
                # only accept whitelisted commands
                if name in {"launch_monitor", "set_device", "noop"}:
                    cmd = {"name": name, "args": args}
    except Exception:
        # On any parse error, fall back to noop
        cmd = default_cmd

    return clean.strip(), cmd


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

    text_only  = _is_truthy(body.get("text_only"))

    # Identity fast-path
    if speaker and re.search(r"\b(who am i|what'?s my name|who'?s speaking)\b", text.lower()):
        plain = {"text": f"You are {speaker}.", "audio": {"audio_base64": None}}
        if text_only:
            return _ok(plain, headers=out_hdr)
        try:
            return _ok(_tts(plain["text"], voice_id), headers=out_hdr)
        except Exception as exc:
            # If Polly fails, still give a 200 with text only to avoid breaking UX on identity
            return _ok(plain, headers=out_hdr)

    # Short memory (optional)
    try:
        history: List[Dict[str, str]] = _get_memory(session_id) if USE_MEMORY else []
    except Exception as exc:
        # Memory failures should not 500 the whole request
        history = []

    system = _system_prompt(speaker, acoustic)

    # Generate reply (LLM)
    try:
        raw_reply = _call_llm(MODEL_ID, system, history, text)
    except Exception as exc:
        # Return structured 502 so the client can see cause immediately
        return _server_error(
            exc,
            headers=out_hdr,
            request_id=getattr(context, "aws_request_id", "unknown"),
            code=502,
        )

    # Extract structured command + clean text
    reply, command = _extract_command(raw_reply)

    # Update memory (best-effort)
    try:
        if USE_MEMORY:
            history.append({"user": text, "assistant": reply})
            _put_memory(session_id, history)
    except Exception:
        pass

    # TTS (Polly) — if Polly fails, report a 502 with details instead of 500
    if text_only:
        return _ok(
            {"text": reply, "command": command, "audio": {"audio_base64": None}},
            headers=out_hdr,
        )
    try:
        tts_payload = _tts(reply, voice_id)
        # propagate command alongside text+audio
        tts_payload["command"] = command
        return _ok(tts_payload, headers=out_hdr)
    except Exception as exc:
        return _server_error(
            exc,
            headers=out_hdr,
            request_id=getattr(context, "aws_request_id", "unknown"),
            code=502,
        )
