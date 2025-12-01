from __future__ import annotations

"""
Text-to-speech helpers using Amazon Polly.

This module provides a small SpeechSynthesizer wrapper that can:

- Turn a text string into spoken audio using a configured Polly voice.
- Optionally upload the audio to an S3 bucket.
- Always return a base64-encoded representation of the audio data so that
  clients can play it directly without touching S3.

The constructor is intentionally simple and may be used both from Lambda
(broker) and from local tooling if AWS credentials are configured.
"""

import base64
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from xml.sax.saxutils import escape

import boto3


class SpeechSynthesizer:
    """Simple Polly-backed text-to-speech engine."""

    def __init__(
        self,
        bucket: Optional[str] = None,
        voice_id: str = "Matthew",
        region_name: Optional[str] = None,
    ) -> None:
        """
        Parameters
        ----------
        bucket:
            Optional S3 bucket name to which synthesized audio will be uploaded.
            If None, audio will not be written to S3 and only a base64-encoded
            payload will be returned.
        voice_id:
            Polly VoiceId to use (e.g., "Matthew", "Joanna", "Amy", etc.).
        region_name:
            AWS region for the Polly (and optional S3) clients. If omitted, the
            AWS_REGION or REGION environment variable is used, falling back to
            "us-west-2".
        """
        region = region_name or os.getenv("AWS_REGION") or os.getenv("REGION") or "us-west-2"
        self._polly = boto3.client("polly", region_name=region)
        self._bucket = bucket
        self._s3 = boto3.client("s3", region_name=region) if bucket else None
        self._voice = voice_id
        self._region = region

    @staticmethod
    def _to_ssml(text: str) -> str:
        """Convert plain text into a minimal SSML document."""
        # Escape special XML characters and replace hard newlines with breaks.
        escaped = escape(text)
        escaped = escaped.replace("\n\n", "<break time='800ms'/>").replace(
            "\n", "<break time='400ms'/>"
        )
        return f"<speak>{escaped}</speak>"

    def synthesize(self, text: str, key_prefix: Optional[str] = None) -> Dict[str, Any]:
        """Synthesize `text` to speech and optionally upload to S3.

        Returns a dict containing at least:

            {
              "audio_base64": "<base64-encoded MP3 bytes>"
            }

        and, when an S3 bucket is configured:

            {
              "bucket": "<bucket-name>",
              "s3_key": "<object-key>",
              "audio_base64": "..."
            }
        """
        if not text:
            raise ValueError("SpeechSynthesizer.synthesize() requires non-empty text")

        ssml = self._to_ssml(text)
        response = self._polly.synthesize_speech(
            VoiceId=self._voice,
            OutputFormat="mp3",
            TextType="ssml",
            Text=ssml,
        )
        audio_stream = response["AudioStream"].read()

        result: Dict[str, Any] = {
            "audio_base64": base64.b64encode(audio_stream).decode("ascii"),
        }

        if self._bucket and self._s3 is not None:
            # Generate a reasonably unique key.
            ts = datetime.now(timezone.utc).isoformat()
            safe_ts = ts.replace(":", "-")
            prefix = (key_prefix or "agent").strip("/")
            key = f"{prefix}/{safe_ts}.mp3"
            self._s3.put_object(
                Bucket=self._bucket,
                Key=key,
                Body=audio_stream,
                ContentType="audio/mpeg",
            )
            result.update({"bucket": self._bucket, "s3_key": key})

        return result


__all__ = ["SpeechSynthesizer"]
