"""
bm25_utils.py - Fast sparse keyword retrieval for hybrid RAG
Uses rank_bm25 for BM25 keyword search over document chunks.
"""
from rank_bm25 import BM25Okapi
import re

class BM25Retriever:
    def __init__(self, chunks):
        # Tokenize chunks for BM25
        self.chunks = chunks
        self.tokenized_chunks = [self._tokenize(c) for c in chunks]
        self.bm25 = BM25Okapi(self.tokenized_chunks)

    def _tokenize(self, text):
        # Simple whitespace and punctuation split
        return re.findall(r'\w+', text.lower())

    def get_top_k(self, query, k=5):
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [(self.chunks[i], scores[i]) for i in top_indices if scores[i] > 0]

# Convenience function for one-off retrieval

def bm25_retrieve(query, chunks, k=5):
    retriever = BM25Retriever(chunks)
    return retriever.get_top_k(query, k=k)
