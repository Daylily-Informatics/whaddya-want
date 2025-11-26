# Lightweight speaker embedding/enrollment using SpeechBrain ECAPA-TDNN.
# Stores .npy embeddings in ~/.whaddya/speakers/{name}.npy

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np

try:
    import torch  # type: ignore
    from speechbrain.pretrained import EncoderClassifier  # type: ignore

    SPEAKER_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - optional dependency
    EncoderClassifier = None  # type: ignore
    torch = None  # type: ignore
    SPEAKER_IMPORT_ERROR = exc
    print("[diag] speechbrain import error:", repr(exc), file=sys.stderr)

SPEAKER_MODEL_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"
SPEAKER_EMBED_DIR = Path(os.path.expanduser("~/.whaddya/speakers"))
SPEAKER_EMBED_DIR.mkdir(parents=True, exist_ok=True)


class SpeakerEmbedder:
    """Lazy-loading wrapper around SpeechBrain speaker embeddings."""

    def __init__(self):
        self.enabled = SPEAKER_IMPORT_ERROR is None
        self.cache: Dict[str, np.ndarray] = {}
        self._model: EncoderClassifier | None = None  # type: ignore[type-arg]
        self._load_all()

    def _load_all(self) -> None:
        for p in SPEAKER_EMBED_DIR.glob("*.npy"):
            self.cache[p.stem] = np.load(p)

    def _ensure_model(self) -> None:
        if not self.enabled or self._model is not None or EncoderClassifier is None:
            return
        self._model = EncoderClassifier.from_hparams(
            source=SPEAKER_MODEL_SOURCE,
            run_opts={"device": "cpu"},
        )

    def embed(self, wav_16k_mono: np.ndarray) -> Optional[np.ndarray]:
        """Compute a speaker embedding for a mono 16 kHz waveform."""

        if not self.enabled:
            return None
        self._ensure_model()
        if self._model is None or torch is None:
            return None
        with torch.no_grad():
            emb = self._model.encode_batch(torch.from_numpy(wav_16k_mono).unsqueeze(0))
        return emb.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)

    def enroll(self, name: str, wav_16k_mono: np.ndarray) -> bool:
        emb = self.embed(wav_16k_mono)
        if emb is None:
            return False
        np.save(SPEAKER_EMBED_DIR / f"{name}.npy", emb)
        self.cache[name] = emb
        return True

    def identify(self, wav_16k_mono: np.ndarray, threshold: float = 0.65) -> Optional[str]:
        if not self.cache:
            return None
        probe = self.embed(wav_16k_mono)
        if probe is None:
            return None
        best, best_name = -1.0, None
        for name, ref in self.cache.items():
            sim = float(np.dot(probe, ref) / (np.linalg.norm(probe) * np.linalg.norm(ref) + 1e-9))
            if sim > best:
                best, best_name = sim, name
        return best_name if best >= threshold else None


__all__ = [
    "SPEAKER_EMBED_DIR",
    "SPEAKER_IMPORT_ERROR",
    "SPEAKER_MODEL_SOURCE",
    "SpeakerEmbedder",
]
