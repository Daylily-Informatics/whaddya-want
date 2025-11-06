# lambda/broker/app.py
from __future__ import annotations
import json, os, base64, traceback, sys
import boto3

REGION = os.getenv("AWS_REGION", "us-west-2")
POLLY_VOICE = os.getenv("POLLY_VOICE", "Joanna")

polly = boto3.client("polly", region_name=REGION)

def _resp(status: int, payload: dict, *, is_b64=False, headers=None):
    return {
        "statusCode": status,
        "headers": {"content-type":"application/json", **(headers or {})},
        "isBase64Encoded": is_b64,
        "body": json.dumps(payload),  # IMPORTANT: body must be a JSON STRING
    }

def handler(event, context):
    # Loud logging so we can see what's happening
    try:
        print("EVENT:", json.dumps(event)[:2000], file=sys.stdout)
    except Exception:
        pass

    try:
        # Parse body (handles base64-encoded proxy requests)
        body = event.get("body") or "{}"
        if event.get("isBase64Encoded"):
            body = base64.b64decode(body).decode("utf-8", "ignore")
        data = json.loads(body)
        text_in = (data.get("text") or "").strip()

        if not text_in:
            return _resp(400, {"error":"missing 'text'"})

        # Minimal reply text (echo)
        reply_text = f"Echo: {text_in}"

        # Tiny Polly synth to prove audio works
        ssml = f"<speak><prosody rate='medium'>{reply_text}</prosody></speak>"
        polly_res = polly.synthesize_speech(
            TextType="ssml",
            Text=ssml,
            VoiceId=POLLY_VOICE,
            OutputFormat="mp3",
        )
        mp3_bytes = polly_res["AudioStream"].read()

        # Return JSON payload with base64 MP3 (proxy-spec body must be JSON string)
        return _resp(200, {
            "text": reply_text,
            "audio": {"audio_base64": base64.b64encode(mp3_bytes).decode("ascii")}
        })

    except Exception as e:
        print("ERROR:", e, file=sys.stderr)
        traceback.print_exc()
        return _resp(500, {"error":"broker error"})
