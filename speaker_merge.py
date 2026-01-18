import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def merge_similar_speakers(avg_embeddings, threshold=0.85):
    speakers = list(avg_embeddings.keys())
    mapping = {}

    for i, s1 in enumerate(speakers):
        if s1 in mapping:
            continue

        mapping[s1] = s1

        for j in range(i+1, len(speakers)):
            s2 = speakers[j]
            sim = cosine_similarity(
                [avg_embeddings[s1]],
                [avg_embeddings[s2]]
            )[0][0]

            if sim > threshold:
                mapping[s2] = s1

    return mapping
