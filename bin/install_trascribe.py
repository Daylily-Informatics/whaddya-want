# stt_stream.py (minimal, mono 16k PCM to AWS)
import asyncio, sys, sounddevice as sd, numpy as np
from amazon_transcribe.eventstream import EventStream
from amazon_transcribe.client import TranscribeStreamingClient

RATE=16000; CH=1; LANG="en-US"

async def main():
    client = TranscribeStreamingClient()
    stream = await client.start_stream_transcription(
        language_code=LANG, media_sample_rate_hz=RATE, media_encoding="pcm"
    )
    async def mic():
        with sd.InputStream(samplerate=RATE, channels=CH, dtype="int16") as s:
            while True:
                audio = s.read(3200)[0].tobytes()  # 0.2s
                await stream.input_stream.send_audio_event(audio_chunk=audio)
    async def printer():
        async for event in stream.output_stream:
            if hasattr(event, "transcript"):
                for r in event.transcript.results:
                    if r.is_partial: continue
                    print("USER:", "".join([a.transcript for a in r.alternatives]))
    await asyncio.gather(mic(), printer())

if __name__ == "__main__":
    asyncio.run(main())
