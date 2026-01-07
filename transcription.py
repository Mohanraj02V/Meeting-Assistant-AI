import whisper
import torch

def transcribe_audio(audio_path: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = whisper.load_model(
        "medium",   # multilingual
        device=device
    )

    result = model.transcribe(
        audio_path,
        task="translate",   # auto → English
        fp16=torch.cuda.is_available()
    )

    return result
