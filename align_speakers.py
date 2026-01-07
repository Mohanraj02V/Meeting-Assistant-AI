def assign_speakers(transcript, speaker_segments):
    final = []

    for seg in transcript["segments"]:
        mid = (seg["start"] + seg["end"]) / 2

        speaker_name = "Unknown"
        for sp in speaker_segments:
            if sp["start"] <= mid <= sp["end"]:
                speaker_name = sp["speaker"]
                break

        final.append({
            "speaker": speaker_name,
            "text": seg["text"]
        })

    return final
