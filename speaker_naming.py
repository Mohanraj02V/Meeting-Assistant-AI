'''
import re

def resolve_speaker_names(segments):
    speaker_map = {}
    unknown_index = 1

    for seg in segments:
        speaker = seg["speaker"]
        text = seg["text"].lower()

        name_match = re.search(
            r"(my name is|i am|this is)\s+([a-z]+)", text
        )

        if name_match:
            speaker_map[speaker] = name_match.group(2).capitalize()

        if speaker not in speaker_map:
            speaker_map[speaker] = f"Unknown Person {unknown_index}"
            unknown_index += 1

        seg["speaker_name"] = speaker_map[speaker]

    return segments
'''

import re

COMMON_VERBS = {
    "doing", "recording", "working", "testing", "speaking",
    "talking", "saying", "going", "making", "creating"
}

def resolve_speaker_names(segments):
    speaker_map = {}
    unknown_index = 1

    for seg in segments:
        raw_speaker = seg["speaker"]
        text = seg["text"]

        # If already resolved, just assign
        if raw_speaker in speaker_map:
            seg["speaker_name"] = speaker_map[raw_speaker]
            continue

        # Try to detect name
        match = re.search(
            r"(?:my name is|i am|this is)\s+([A-Z][a-z]{2,})",
            text,
            re.IGNORECASE
        )

        if match:
            name = match.group(1).capitalize()

            # Reject verbs pretending to be names
            if name.lower() not in COMMON_VERBS:
                speaker_map[raw_speaker] = name
                seg["speaker_name"] = name
                continue

        # Otherwise assign Unknown
        speaker_map[raw_speaker] = f"Unknown Person {unknown_index}"
        seg["speaker_name"] = speaker_map[raw_speaker]
        unknown_index += 1

    return segments
