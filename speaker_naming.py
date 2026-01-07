import re

def resolve_speaker_names(segments):
    speaker_map = {}
    unknown_index = 1

    for seg in segments:
        speaker = seg["speaker"]
        text = seg["text"].lower()

        name_match = re.search(
            r"(my name is|i am)\s+([a-z]+)", text
        )

        if name_match:
            speaker_map[speaker] = name_match.group(2).capitalize()

        if speaker not in speaker_map:
            speaker_map[speaker] = f"Unknown Person {unknown_index}"
            unknown_index += 1

        seg["speaker_name"] = speaker_map[speaker]

    return segments
