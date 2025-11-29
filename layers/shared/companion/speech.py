"""Text-to-speech helpers using Amazon Polly."""
from __future__ import annotations

import base64
from xml.sax.saxutils import escape

import boto3


class SpeechSynthesizer:
    def __init__(self, bucket: str, voice_id: str, region_name: str) -> None:
        self._polly = boto3.client("polly", region_name=region_name)
        self._s3 = boto3.client("s3", region_name=region_name)
        self._bucket = bucket
        self._voice = voice_id

    def synthesize(
        self,
        text: str,
        session_id: str,
        response_id: str,
        *,
        voice_id: str | None = None,
    ) -> dict[str, str]:
        """Synthesize speech to MP3, store in S3, and return playback metadata.

        The returned mapping has keys:
          - bucket: S3 bucket name
          - s3_key: object key under that bucket
          - audio_base64: MP3 data encoded as base64 for inline playback
        """
        ssml_text = f'<speak><prosody rate="slow" pitch="-6%">{escape(text)}</prosody></speak>'
        voice = voice_id or self._voice
        polly = self._polly.synthesize_speech(
            Text=ssml_text,
            TextType="ssml",
            VoiceId=voice,
            OutputFormat="mp3",
        )
        stream_body = polly["AudioStream"]
        try:
            audio_stream = stream_body.read()
        finally:  # pragma: no cover - best effort clean-up
            stream_body.close()
        key = f"responses/{session_id}/{response_id}.mp3"
        self._s3.put_object(
            Bucket=self._bucket,
            Key=key,
            Body=audio_stream,
            ContentType="audio/mpeg",
        )
        return {
            "s3_key": key,
            "bucket": self._bucket,
            "audio_base64": base64.b64encode(audio_stream).decode("ascii"),
        }


__all__ = ["SpeechSynthesizer"]
