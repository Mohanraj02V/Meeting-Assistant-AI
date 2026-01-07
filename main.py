from audio_preprocess import preprocess_audio
from transcription import transcribe_audio
from diarization import diarize_audio
from align_speakers import assign_speakers
from speaker_naming import resolve_speaker_names
from intelligence import analyze_meeting

INPUT_AUDIO = "project.mp3"
CLEAN_AUDIO = "clean.wav"

# 1️⃣ Clean audio
preprocess_audio(INPUT_AUDIO, CLEAN_AUDIO)

# 2️⃣ Transcribe
transcript = transcribe_audio(CLEAN_AUDIO)

# 3️⃣ Diarize
speaker_segments = diarize_audio(CLEAN_AUDIO)

# 4️⃣ Align speakers
segments = assign_speakers(transcript, speaker_segments)

# 5️⃣ Resolve names
segments = resolve_speaker_names(segments)

# 6️⃣ Final transcript
final_text = "\n".join(
    f"{s['speaker_name']}: {s['text']}" for s in segments
)

print("\n===== TRANSCRIPT =====\n")
print(final_text)

# 7️⃣ Intelligence
analysis = analyze_meeting(final_text)

print("\n===== MEETING INSIGHTS =====\n")
print(analysis)
