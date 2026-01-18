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

import os
import numpy as np
import torch
from pyannote.audio import Pipeline
from pyannote.audio.core.inference import Inference

# -------------------------------------------------------------------
# Environment & device
# -------------------------------------------------------------------
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

if not HF_TOKEN:
    raise RuntimeError(
        "HUGGINGFACE_TOKEN is missing. "
        "Add it in Render → Environment Variables."
    )

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------------------------------------------
# Load models (AUTH + DEVICE ARE MANDATORY)
# -------------------------------------------------------------------
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization",
    use_auth_token=HF_TOKEN
)
pipeline.to(DEVICE)

embedding_model = Inference(
    "pyannote/embedding",
    window="whole",
    use_auth_token=HF_TOKEN,
    device=DEVICE
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

        # Pyannote expects a dict input
        emb = embedding_model({
            "audio": audio_path,
            "start": turn.start,
            "end": turn.end,
        })

        embeddings.setdefault(speaker, []).append(emb)

    # Average embeddings per speaker
    avg_embeddings = {
        speaker: np.mean(v, axis=0)
        for speaker, v in embeddings.items()
    }

    return speakers, avg_embeddings
