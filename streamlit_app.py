import streamlit as st
import os
import tempfile

from audio_preprocess import preprocess_audio
from transcription import transcribe_audio
from diarization import diarize_audio
from align_speakers import assign_speakers
from speaker_naming import resolve_speaker_names
from intelligence import analyze_meeting
from speaker_merge import merge_similar_speakers

st.set_page_config(page_title="Meeting Assistant AI", layout="wide")

st.title("🎙️ AI Meeting Assistant")
st.caption("Upload an audio file and get speaker-wise transcript + meeting intelligence")

# ---------------------------
# Upload section
# ---------------------------
uploaded_file = st.file_uploader(
    "Upload meeting audio (MP3, WAV, M4A, FLAC)",
    type=["mp3", "wav", "m4a", "flac"]
)

if uploaded_file:
    with tempfile.TemporaryDirectory() as temp_dir:
        input_audio = os.path.join(temp_dir, uploaded_file.name)
        clean_audio = os.path.join(temp_dir, "clean.wav")

        # Save uploaded file
        with open(input_audio, "wb") as f:
            f.write(uploaded_file.read())

        col1, col2 = st.columns([1, 1])

        with col1:
            st.audio(input_audio)

        if st.button("🚀 Process Meeting"):
            progress = st.progress(0)
            status = st.empty()

            # -------------------------------------
            # Step 1 — Preprocess
            # -------------------------------------
            status.text("🔹 Cleaning audio...")
            preprocess_audio(input_audio, clean_audio)
            progress.progress(15)

            # -------------------------------------
            # Step 2 — Transcription
            # -------------------------------------
            status.text("🔹 Transcribing audio...")
            transcript = transcribe_audio(clean_audio)
            progress.progress(40)

            # -------------------------------------
            # Step 3 — Diarization
            # -------------------------------------
            status.text("🔹 Performing speaker diarization...")
            speaker_segments, embeddings = diarize_audio(clean_audio)
            progress.progress(60)

            # -------------------------------------
            # Step 4 — Speaker merging
            # -------------------------------------
            status.text("🔹 Merging same-voice speakers...")
            merge_map = merge_similar_speakers(embeddings)

            for s in speaker_segments:
                s["speaker"] = merge_map.get(s["speaker"], s["speaker"])

            progress.progress(70)

            # -------------------------------------
            # Step 5 — Align speakers
            # -------------------------------------
            status.text("🔹 Aligning transcript with speakers...")
            segments = assign_speakers(transcript, speaker_segments)
            progress.progress(80)

            # -------------------------------------
            # Step 6 — Resolve names
            # -------------------------------------
            status.text("🔹 Resolving speaker names...")
            segments = resolve_speaker_names(segments)
            progress.progress(90)

            # -------------------------------------
            # Final transcript
            # -------------------------------------
            final_text = "\n".join(
                f"{s['speaker_name']}: {s['text'].strip()}" for s in segments
            )

            progress.progress(95)

            # -------------------------------------
            # Intelligence
            # -------------------------------------
            status.text("🔹 Running meeting intelligence...")
            analysis = analyze_meeting(final_text)
            progress.progress(100)
            status.text("✅ Processing complete")

            # -------------------------------------
            # Output UI
            # -------------------------------------
            st.divider()
            colA, colB = st.columns([1.3, 1])

            with colA:
                st.subheader("🗣️ Speaker-wise Transcript")
                st.text_area(
                    "Transcript",
                    final_text,
                    height=450
                )

            with colB:
                st.subheader("📊 Meeting Insights")
                st.markdown(analysis)
