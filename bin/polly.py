# tts_polly.py
import boto3
text = "Hi, I'm Forge."
polly = boto3.client("polly", region_name="us-east-1")
resp = polly.synthesize_speech(Text=text, VoiceId="Joanna", Engine="neural",
                               OutputFormat="mp3")
with open("reply.mp3","wb") as f: f.write(resp["AudioStream"].read())
# macOS playback:
# !afplay reply.mp3
