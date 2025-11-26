"""Multimodal vision helpers using Anthropic Claude on Amazon Bedrock."""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from typing import Any, Dict, Optional

import boto3


@dataclass(slots=True)
class VisionConfig:
    """Configuration for the Bedrock vision client."""

    region_name: str
    model_id: str


class VisionClient:
    """Thin wrapper around Bedrock Runtime for single-frame scene descriptions."""

    def __init__(self, cfg: VisionConfig) -> None:
        self._cfg = cfg
        self._brt = boto3.client("bedrock-runtime", region_name=cfg.region_name)

    def describe_scene(
        self,
        image_bytes: bytes,
        *,
        hint: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Call a Claude 3/3.5/3.7 Sonnet-like model with an image and get back a compact JSON scene.

        The prompt instructs the model to return a single JSON object with keys:
          - caption: str
          - people:  [ { "approx_count": int, "notes": str } ]
          - animals: [ { "species": str, "approx_count": int, "notes": str } ]
          - objects: [str]
          - layout:  str
        """
        img_b64 = base64.b64encode(image_bytes).decode("ascii")

        user_text = (
            "You are Marvin's camera. Given this single indoor camera frame, "
            "describe the scene as ONE JSON object and nothing else.\n"
            'Schema: {"caption": str, '
            '"people": [{"approx_count": int, "notes": str}], '
            '"animals": [{"species": str, "approx_count": int, "notes": str}], '
            '"objects": [str], "layout": str}.\n'
            "Do not include markdown, backticks, or any extra commentary."
        )
        if hint:
            user_text += f"\nAdditional non-visual hints from other sensors: {hint}"

        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 400,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": img_b64,
                            },
                        },
                        {"type": "text", "text": user_text},
                    ],
                }
            ],
        }

        resp = self._brt.invoke_model(
            modelId=self._cfg.model_id,
            body=json.dumps(body),
            contentType="application/json",
            accept="application/json",
        )
        payload = json.loads(resp["body"].read())

        # Claude Messages API on Bedrock: response["content"] is a list of blocks.
        text_parts: list[str] = []
        for part in payload.get("content", []):
            if isinstance(part, dict) and part.get("type") == "text":
                t = part.get("text")
                if isinstance(t, str):
                    text_parts.append(t)
        text = "".join(text_parts).strip()

        if not text:
            # If the model gave us something unexpected, surface the raw payload.
            return {"raw": payload}

        try:
            scene = json.loads(text)
            if isinstance(scene, dict):
                return scene
            return {"parsed": scene}
        except Exception:
            # Model ignored the JSON-only instruction; return both raw text and payload.
            return {"raw_text": text, "raw": payload}


__all__ = ["VisionClient", "VisionConfig"]
