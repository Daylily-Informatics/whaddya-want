# companion.py (MVP loop: STT -> LLM -> TTS, plus periodic face snapshot)
import asyncio, json, time, subprocess, os, boto3
from amazon_transcribe.client import TranscribeStreamingClient
import sounddevice as sd

REG="us-east-1"; RATE=16000; COL="companion-people"; BUCKET=os.environ["BUCKET"]
brt=boto3.client("bedrock-runtime",region_name=REG)
rk =boto3.client("rekognition",region_name=REG)
pl =boto3.client("polly",region_name=REG)

MODEL="anthropic.claude-3-7-sonnet-2025-05-xx"  # set from list_foundation_models

def take_frame():
    subprocess.run(["imagesnap","-w","1","frame.jpg"], check=True)
    with open("frame.jpg","rb") as f:
        b=f.read()
    try:
        m=rk.search_faces_by_image(CollectionId=COL, Image={"Bytes":b},
                                   FaceMatchThreshold=90, MaxFaces=1)
        who = m["FaceMatches"][0]["Face"]["ExternalImageId"] if m["FaceMatches"] else "unknown"
    except Exception:
        who="unknown"
    # pet species (not identity)
    labels=rk.detect_labels(Image={"Bytes":b}, MaxLabels=5, MinConfidence=80)
    species=[l["Name"] for l in labels["Labels"] if l["Name"] in ("Cat","Dog")]
    return who, species

async def stream_stt_once(seconds=10):
    client=TranscribeStreamingClient()
    stream = await client.start_stream_transcription(
        language_code="en-US", media_sample_rate_hz=RATE, media_encoding="pcm"
    )
    out=[]
    async def mic():
        with sd.InputStream(samplerate=RATE, channels=1, dtype="int16") as s:
            end=time.time()+seconds
            while time.time()<end:
                audio=s.read(3200)[0].tobytes()
                await stream.input_stream.send_audio_event(audio_chunk=audio)
        await stream.input_stream.end_stream()
    async def sink():
        async for e in stream.output_stream:
            if hasattr(e,"transcript"):
                for r in e.transcript.results:
                    if not r.is_partial:
                        out.append(" ".join(a.transcript for a in r.alternatives))
    await asyncio.gather(mic(), sink())
    return " ".join(out).strip()

def llm(prompt, who, species):
    sysmsg="You are Forge. Be concise. If 'actions' needed, return JSON: {\"action\":\"open_url\",\"url\":\"...\"}."
    body={
      "anthropic_version":"bedrock-2023-05-31",
      "max_tokens":400,
      "system":sysmsg,
      "messages":[{"role":"user","content":[{"type":"text","text":
        f"[vision] person={who}, pets={species}\n[heard] {prompt}"}]}]
    }
    r=brt.invoke_model(modelId=MODEL, body=json.dumps(body))
    j=json.loads(r["body"].read())
    # Anthropic on Bedrock returns a messages-like structure:
    text = "".join(p.get("text","") for c in j.get("output",[]) for p in c.get("content",[]))
    return text

def speak(text):
    r=pl.synthesize_speech(Text=text, VoiceId="Joanna", Engine="neural", OutputFormat="mp3")
    with open("reply.mp3","wb") as f: f.write(r["AudioStream"].read())
    subprocess.run(["afplay","reply.mp3"])

async def main():
    print("Speak for ~10s...")
    utter = await stream_stt_once(10)
    who, species = take_frame()
    reply = llm(utter or "No speech detected.", who, species)
    print("FORGE:", reply)
    speak(reply)

if __name__=="__main__":
    asyncio.run(main())
