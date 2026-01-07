import os
import subprocess
import tempfile
import librosa
import noisereduce as nr
import soundfile as sf
import shutil

# 🔥 ABSOLUTE PATH to OFFICIAL FFmpeg (NOT conda)
FFMPEG_PATH = r"D:\ffmpeg-2025-12-28-git-9ab2a437a1-full_build\ffmpeg-2025-12-28-git-9ab2a437a1-full_build\bin\ffmpeg.exe"


def _check_ffmpeg():
    """Ensure FFmpeg exists and is callable"""
    if not os.path.exists(FFMPEG_PATH):
        raise RuntimeError(f"FFmpeg not found at: {FFMPEG_PATH}")

    try:
        subprocess.run(
            [FFMPEG_PATH, "-version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True
        )
    except Exception:
        raise RuntimeError("FFmpeg is not executable or crashed.")


def decode_to_wav(input_path: str, wav_path: str):
    """
    Decode ANY audio format to clean PCM 16-bit WAV
    Uses OFFICIAL FFmpeg only (Windows-safe)
    """
    cmd = [
        FFMPEG_PATH,
        "-y",
        "-loglevel", "error",
        "-i", input_path,
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-acodec", "pcm_s16le",
        wav_path
    ]

    subprocess.run(cmd, check=True)


def preprocess_audio(input_path: str, output_path: str) -> str:
    """
    FINAL preprocessing pipeline:
    - MP3 / WAV / M4A / FLAC / AAC supported
    - Noise reduction
    - Whisper & pyannote safe output
    """

    _check_ffmpeg()

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # --------------------------------------------------
    # STEP 1: Decode input audio → temp WAV
    # --------------------------------------------------
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        decoded_wav = tmp.name

    try:
        decode_to_wav(input_path, decoded_wav)

        # --------------------------------------------------
        # STEP 2: Load decoded WAV safely
        # --------------------------------------------------
        audio, sr = librosa.load(
            decoded_wav,
            sr=16000,
            mono=True
        )

        if audio.size == 0:
            raise RuntimeError("Decoded audio is empty")

        # --------------------------------------------------
        # STEP 3: Noise reduction
        # --------------------------------------------------
        reduced = nr.reduce_noise(
            y=audio,
            sr=sr,
            stationary=True
        )

        # --------------------------------------------------
        # STEP 4: Write FINAL Whisper-safe WAV
        # --------------------------------------------------
        sf.write(
            output_path,
            reduced,
            sr,
            subtype="PCM_16"
        )

    finally:
        # --------------------------------------------------
        # CLEANUP (important on Windows)
        # --------------------------------------------------
        if os.path.exists(decoded_wav):
            try:
                os.remove(decoded_wav)
            except PermissionError:
                pass  # Windows file lock safety

    return output_path
