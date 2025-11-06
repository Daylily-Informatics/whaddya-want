# lambda/broker/app.py
import os, json, base64, re, time
from typing import List, Dict, Any, Optional

import boto3
from botocore.config import Config

# ----- Env -----
REGION: str = os.environ.get("AWS_REGION", "us-west-2")
POLLY_VOICE: str = os.environ.get("POLLY_VOICE", "Joanna")

# Default to Titan since that's what you're using. You can change at runtime via env update.
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


# ----- Utils -----
def _ok(body: Dict[str, Any]) -> Dict[str, Any]:
    return {"statusCode": 200, "headers": {"content-type": "application/json"}, "body": json.dumps(body)}

def _tts(text: str, voice_id: Optional[str] = None) -> Dict[str, Any]:
    voice = voice_id or POLLY_VOICE
    r = polly.synthesize_speech(Text=text, VoiceId=voice, OutputFormat="mp3")
    audio = r["AudioStream"].read()
    return {"text": text, "audio": {"audio_base64": base64.b64encode(audio).decode()}}

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

def _system_prompt(speaker: Optional[str], acoustic_event: Optional[str]) -> str:
    # NOTE: Some models (e.g., Titan Text Express) don't support system messages in Converse,
    # so we only use this when the model supports it (or we embed it into the prompt text).
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

def _parse_body(event: Dict[str, Any]) -> Dict[str, Any]:
    body = event.get("body")
    if isinstance(body, str):
        if event.get("isBase64Encoded"):
            try:
                body = base64.b64decode(body).decode("utf-8")
            except Exception:
                pass
        try:
            body = json.loads(body)
        except Exception:
            body = {}
    return body or {}


# ----- LLM calls -----
def _call_titan_text_express(system: str, history: List[Dict[str, str]], user_text: str) -> str:
    """
    Titan Text Express (amazon.titan-text-express-v1) via InvokeModel.
    Schema:
      {
        "inputText": "...",
        "textGenerationConfig": {
           "maxTokenCount": int, "temperature": float, "topP": float, "stopSequences": [str,...]
        }
      }
    """
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
    """
    Model-agnostic Bedrock Converse call.
    Some models don't support 'system' messages; in that case we retry without 'system'.
    """
    def _msgs():
        msgs: List[Dict[str, Any]] = []
        for t in history:
            msgs.append({"role": "user", "content": [{"text": t["user"]}]})
            msgs.append({"role": "assistant", "content": [{"text": t["assistant"]}]})
        msgs.append({"role": "user", "content": [{"text": user_text or "Respond briefly."}]})
        return msgs

    # First try WITH system, then retry WITHOUT if the model complains.
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
            # Retry without system
            kwargs.pop("system", None)
            out = brt.converse(**kwargs)
            return out["output"]["message"]["content"][0]["text"]
        raise

def _call_llm(model_id: str, system: str, history: List[Dict[str, str]], user_text: str) -> str:
    """
    Route Titan Text Express through InvokeModel; others through Converse (with system if supported).
    If you later set MODEL_ID to an inference-profile ARN (e.g., Nova), this will use Converse correctly.
    """
    if model_id.startswith("amazon.titan-text-express"):
        return _call_titan_text_express(system, history, user_text)
    return _call_converse(model_id, system, history, user_text)


# ----- Lambda handler -----
def handler(event, _ctx):
    body = _parse_body(event)

    session_id = (body.get("session_id") or "anon").strip() or "anon"
    text       = (body.get("text") or "").strip()
    context    = body.get("context") or {}
    speaker    = context.get("speaker_id")
    acoustic   = context.get("acoustic_event")
    voice_id   = body.get("voice_id")
    if isinstance(voice_id, str):
        voice_id = voice_id.strip() or None
    else:
        voice_id = None

    # Identity fast-path
    if speaker and re.search(r"\b(who am i|what'?s my name|who'?s speaking)\b", text.lower()):
        return _ok(_tts(f"You are {speaker}.", voice_id))

    # Short memory (optional)
    history: List[Dict[str, str]] = _get_memory(session_id) if USE_MEMORY else []

    system = _system_prompt(speaker, acoustic)

    try:
        reply = _call_llm(MODEL_ID, system, history, text)
    except Exception as e:
        # Never hard-fail the request
        reply = f"{(speaker + ': ') if speaker else ''}I heard you say: {text or '<silence>'}. (LLM unavailable: {e})"

    if USE_MEMORY:
        history.append({"user": text, "assistant": reply})
        _put_memory(session_id, history)

    return _ok(_tts(reply, voice_id))
