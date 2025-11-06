# Lightweight speaker embedding/enrollment using SpeechBrain ECAPA-TDNN.
# Stores .npy embeddings in ~/.whaddya/speakers/{name}.npy

from __future__ import annotations
import os, pathlib, numpy as np, torch
from speechbrain.pretrained import EncoderClassifier
from typing import Optional

_SPK_DIR = pathlib.Path(os.path.expanduser("~/.whaddya/speakers"))
_SPK_DIR.mkdir(parents=True, exist_ok=True)

class SpeakerID:
    def __init__(self):
        self.model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": "cpu"}
        )
        self.cache = {}  # name -> embedding vector
        self._load_all()

    def _load_all(self):
        for p in _SPK_DIR.glob("*.npy"):
            self.cache[p.stem] = np.load(p)

    def _embed(self, wav_16k_mono: np.ndarray) -> np.ndarray:
        # expects float32 [-1,1] at 16 kHz
        with torch.no_grad():
            emb = self.model.encode_batch(torch.from_numpy(wav_16k_mono).unsqueeze(0))
        return emb.squeeze(0).squeeze(0).cpu().numpy()

    def enroll(self, name: str, wav_16k_mono: np.ndarray):
        e = self._embed(wav_16k_mono)
        np.save(_SPK_DIR / f"{name}.npy", e)
        self.cache[name] = e

    def identify(self, wav_16k_mono: np.ndarray, threshold: float = 0.65) -> Optional[str]:
        if not self.cache:
            return None
        probe = self._embed(wav_16k_mono)
        # cosine similarity
        best, best_name = -1.0, None
        for name, ref in self.cache.items():
            sim = float(np.dot(probe, ref) / (np.linalg.norm(probe) * np.linalg.norm(ref) + 1e-9))
            if sim > best:
                best, best_name = sim, name
        return best_name if best >= threshold else None
