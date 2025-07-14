from sentence_transformers import CrossEncoder
import numpy as np

# Download and cache cross-encoder model on first use
_CROSS_ENCODER_MODEL = None
CROSS_ENCODER_NAME = 'cross-encoder/ms-marco-MiniLM-L-6-v2'

def get_cross_encoder():
    global _CROSS_ENCODER_MODEL
    if _CROSS_ENCODER_MODEL is None:
        _CROSS_ENCODER_MODEL = CrossEncoder(CROSS_ENCODER_NAME)
    return _CROSS_ENCODER_MODEL

def rerank_chunks(query, chunks, top_k=2):
    """
    Re-rank text chunks by true semantic relevance to the query using a cross-encoder.
    Returns the top_k most relevant chunks.
    """
    if not chunks:
        return []
    model = get_cross_encoder()
    pairs = [[query, chunk] for chunk in chunks]
    scores = model.predict(pairs)
    ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
    return [chunk for chunk, score in ranked[:top_k]]
