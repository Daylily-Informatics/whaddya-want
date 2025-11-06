"""Utilities for lightweight sound classification metadata."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

_DOG_KEYWORDS = {"woof", "bark", "ruff", "arf", "bow wow", "grr", "growl"}


@dataclass(slots=True)
class SoundClassification:
    """Represents coarse sound classification metadata."""

    speaker_label: str | None
    sound_type: str
    confidence: float
    context_message: str | None

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "sound_type": self.sound_type,
            "confidence": round(self.confidence, 3),
        }
        if self.speaker_label:
            payload["speaker_label"] = self.speaker_label
        if self.context_message:
            payload["context_message"] = self.context_message
        return payload


class SoundClassifier:
    """Best-effort sound classification using lightweight heuristics.

    The client may provide hints such as a speaker label (from Amazon
    Transcribe's speaker separation) or a pre-computed ``sound_type``.
    When hints are missing we perform small text-based heuristics to
    guess whether a sound resembles a dog bark versus a human speaker.
    """

    def classify(
        self,
        transcript: str,
        *,
        speaker_label: str | None = None,
        sound_type: str | None = None,
    ) -> SoundClassification:
        normalized = (transcript or "").strip().lower()
        if sound_type:
            derived_type = sound_type
            confidence = 0.6
        elif speaker_label:
            derived_type = "human"
            confidence = 0.75
        elif self._looks_like_dog(normalized):
            derived_type = "dog"
            confidence = 0.65
        elif normalized:
            derived_type = "human"
            confidence = 0.4
        else:
            derived_type = "unknown"
            confidence = 0.2

        context_message = self._build_context_message(
            derived_type=derived_type, speaker_label=speaker_label
        )
        return SoundClassification(
            speaker_label=speaker_label,
            sound_type=derived_type,
            confidence=confidence,
            context_message=context_message,
        )

    @staticmethod
    def _looks_like_dog(text: str) -> bool:
        if not text:
            return False
        for keyword in _DOG_KEYWORDS:
            if keyword in text:
                return True
        # Single syllable repeated sounds such as "woof woof"
        parts = text.split()
        if parts and len(parts) <= 5:
            unique = {part for part in parts}
            if len(unique) <= 2 and all(len(part) <= 5 for part in unique):
                return True
        return False

    @staticmethod
    def _build_context_message(
        *, derived_type: str, speaker_label: str | None
    ) -> str | None:
        if derived_type == "human":
            if speaker_label:
                return f"The most recent utterance came from speaker '{speaker_label}'."
            return "The most recent audio appeared to come from a human speaker."
        if derived_type == "dog":
            return "The detected sound resembled a dog bark."
        if derived_type == "unknown":
            return "The source of the last audio clip could not be identified."
        return None


__all__ = ["SoundClassifier", "SoundClassification"]
