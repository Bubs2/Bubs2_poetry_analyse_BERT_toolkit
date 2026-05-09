import numpy as np

from enum import Enum
from embedding import Embedding
from split_poem import SplitMode, split_poem


class ExtractMode(Enum):
    Span = 0
    Prompt = 1


def get_all_segment_vectors(
        emb: Embedding,
        extract_mode: ExtractMode,
        split_mode: SplitMode,
        context
):
    segments = split_poem(context, split_mode)

    segment_texts = []
    vectors = []

    for seg_text, start, end in segments:
        if extract_mode == ExtractMode.Span:
            vec = emb.get_vector(context, (start, end))
        else:
            vec = emb.get_vector(f"{context}中的{'行' if split_mode == SplitMode.Line else '段'}：{seg_text}")
        segment_texts.append(seg_text)
        vectors.append(vec)

    return segment_texts, np.array(vectors)