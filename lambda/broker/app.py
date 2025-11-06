import os, json, base64, re, time
import boto3
from botocore.config import Config

REGION = os.environ.get("AWS_REGION", "us-west-2")
POLLY_VOICE = os.environ.get("POLLY_VOICE", "Joanna")
MODEL_ID = os.environ.get("MODEL_ID", "anthropic.claude-3-5-sonnet-20241022-v2:0")
HISTORY_LIMIT = int(os.environ.get("HISTORY_LIMIT", "12"))
USE_MEMORY = os.environ.get("USE_MEMORY", "false").lower() == "true"
TABLE = os.environ.get("CONVERSATION_TABLE")  # optional; only used if USE_MEMORY=true

cfg = Config(retries={"max_attempts": 3, "mode": "standard"})
polly = boto3.client("polly", region_name=REGION, config=cfg)
bedrock = boto3.client("bedrock-runtime", region_name=REGION, config=cfg)

dynamodb = boto3.resource("dynamodb", region_name=REGION) if USE_MEMORY and TABLE else None
mem_table = dynamodb.Table(TABLE) if dynamodb else None

def _ok(body: dict):
    return {"statusCode": 200, "headers": {"content-type":"application/json"}, "body": json.dumps(body)}

def _tts(text: str) -> dict:
    r = polly.synthesize_speech(Text=text, VoiceId=POLLY_VOICE, OutputFormat="mp3")
    audio = r["AudioStream"].read()
    return {"text": text, "audio": {"audio_base64": base64.b64encode(audio).decode()}}

def _get_memory(session_id: str):
    if not mem_table: return []
    item = mem_table.get_item(Key={"session_id": session_id}).get("Item")
    return item.get("turns", []) if item else []

def _put_memory(session_id: str, turns):
    if not mem_table: return
    turns = turns[-HISTORY_LIMIT:]
    mem_table.put_item(Item={"session_id": session_id, "turns": turns, "ttl": int(time.time()) + 7*24*3600})

def _system_prompt(speaker: str|None, acoustic_event: str|None) -> str:
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

def _call_bedrock(model_id: str, system: str, history: list[dict], user_text: str) -> str:
    messages = []
    for t in history:
        messages.append({"role":"user","content":[{"type":"text","text": t["user"]}]})
        messages.append({"role":"assistant","content":[{"type":"text","text": t["assistant"]}]})
    messages.append({"role":"user","content":[{"type":"text","text": user_text}]})
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "system": system,
        "messages": messages,
        "max_tokens": 300,
        "temperature": 0.4,
    }
    resp = bedrock.invoke_model(modelId=model_id, body=json.dumps(body))
    out = json.loads(resp["body"].read())
    return out["content"][0]["text"]

def handler(event, _ctx):
    # Parse body
    body = event.get("body")
    if isinstance(body, str):
        try: body = json.loads(body)
        except Exception: body = {}
    body = body or {}

    session_id = body.get("session_id") or "anon"
    text       = (body.get("text") or "").strip()
    context    = body.get("context") or {}
    speaker    = context.get("speaker_id")
    acoustic   = context.get("acoustic_event")

    # Fast-path identity Q
    if speaker and re.search(r"\b(who am i|what'?s my name|who'?s speaking)\b", text.lower()):
        return _ok(_tts(f"You are {speaker}."))

    # Memory (optional)
    history = _get_memory(session_id) if USE_MEMORY else []

    # Conversational reply
    system = _system_prompt(speaker, acoustic)
    try:
        reply = _call_bedrock(MODEL_ID, system, history, text)
    except Exception as e:
        # Fallback so you always get something
        reply = f"{speaker + ': ' if speaker else ''}I heard you say: {text}. (LLM unavailable: {e})"

    # Persist memory
    if USE_MEMORY:
        history.append({"user": text, "assistant": reply})
        _put_memory(session_id, history)

    return _ok(_tts(reply))
