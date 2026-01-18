'''
from pyannote.audio import Pipeline

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization"
)

def diarize_audio(audio_path):
    diarization = pipeline(audio_path)
    speakers = []

    for turn, _, speaker in diarization.itertracks(yield_label=True):
        speakers.append({
            "speaker": speaker,
            "start": float(turn.start),
            "end": float(turn.end)
        })

    return speakers
'''
'''
from pyannote.audio import Pipeline, Inference
import numpy as np

pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")
embedding_model = Inference("pyannote/embedding", window="whole")

def diarize_audio(audio_path):
    diarization = pipeline(audio_path)

    speakers = []
    embeddings = {}

    for turn, _, speaker in diarization.itertracks(yield_label=True):
        speakers.append({
            "speaker": speaker,
            "start": float(turn.start),
            "end": float(turn.end)
        })

        # Correct way: give pyannote a dict
        emb = embedding_model({
            "audio": audio_path,
            "start": turn.start,
            "end": turn.end
        })

        embeddings.setdefault(speaker, []).append(emb)

    # Average embeddings
    avg_embeddings = {
        s: np.mean(v, axis=0) for s, v in embeddings.items()
    }

    return speakers, avg_embeddings
'''

# -------------------------------------------------------------------
# PyTorch 2.6+ compatibility fix for pyannote.audio
# This MUST come before importing pyannote
# -------------------------------------------------------------------
import torch

_original_torch_load = torch.load

def torch_load_compat(*args, **kwargs):
    kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)

torch.load = torch_load_compat

# -------------------------------------------------------------------
# Normal imports
# -------------------------------------------------------------------
import os
import numpy as np
from pyannote.audio import Pipeline
from pyannote.audio.core.inference import Inference

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------
HF_TOKEN = os.environ.get("HUGGINGFACE_TOKEN")

if HF_TOKEN is None:
    raise RuntimeError("HUGGINGFACE_TOKEN environment variable is not set")

DEVICE = torch.device("cpu")

# -------------------------------------------------------------------
# Load models
# -------------------------------------------------------------------
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization",
    use_auth_token=HF_TOKEN,
).to(DEVICE)

embedding_model = Inference(
    "pyannote/embedding",
    window="whole",
    device=DEVICE,
)

# -------------------------------------------------------------------
# Diarization function
# -------------------------------------------------------------------
def diarize_audio(audio_path: str):
    diarization = pipeline(audio_path)

    speakers = []
    embeddings = {}

    for turn, _, speaker in diarization.itertracks(yield_label=True):
        speakers.append({
            "speaker": speaker,
            "start": float(turn.start),
            "end": float(turn.end),
        })

        emb = embedding_model({
            "audio": audio_path,
            "start": turn.start,
            "end": turn.end,
        })

        embeddings.setdefault(speaker, []).append(emb)

    avg_embeddings = {
        speaker: np.mean(vectors, axis=0)
        for speaker, vectors in embeddings.items()
    }

    return speakers, avg_embeddings
