# YAMNet-based acoustic event tagger; we just report 'dog_bark' if present.
import numpy as np, tensorflow as tf, tensorflow_hub as hub, tensorflow_io as tfio

# Class list for YAMNet
_CLASS_MAP = None

def _load_labels():
    global _CLASS_MAP
    if _CLASS_MAP is not None: return
    # YAMNet's class map is bundled with the model; use the TF Hub asset
    labels_path = hub.resolve("https://tfhub.dev/google/yamnet/1") + "/assets/yamnet_class_map.csv"
    raw = tf.io.read_file(labels_path).numpy().decode().splitlines()
    _CLASS_MAP = [r.split(",")[-1] for r in raw]  # display_name column

class BarkDetector:
    def __init__(self):
        _load_labels()
        self.model = hub.load("https://tfhub.dev/google/yamnet/1")

    def detect(self, wav_16k_mono: np.ndarray, prob_threshold: float = 0.25) -> bool:
        waveform = tf.convert_to_tensor(wav_16k_mono, dtype=tf.float32)
        scores, _, _ = self.model(waveform)
        mean_scores = tf.reduce_mean(scores, axis=0).numpy()
        # YAMNet label is 'Dog' and 'Bark' (two labels); accept either
        labels = {lbl: mean_scores[i] for i, lbl in enumerate(_CLASS_MAP)}
        return max(labels.get("Dog", 0.0), labels.get("Bark", 0.0)) >= prob_threshold
