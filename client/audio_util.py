# client/audio_util.py
import numpy as np
def to_16k_mono_float32(wav, sr):
    if sr != 16000:
        # naive linear resample to 16k; for production use torchaudio or librosa
        import librosa
        wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
    if wav.ndim > 1:
        wav = np.mean(wav, axis=1)
    wav = wav.astype(np.float32)
    wav = wav / (np.max(np.abs(wav)) + 1e-9)
    return wav
