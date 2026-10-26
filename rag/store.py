"""Chunking + local embeddings + simple in-memory vector search."""
import numpy as np

_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("all-MiniLM-L6-v2")  # free, runs locally
    return _model


def chunk_text(text: str, size: int = 1000, overlap: int = 200):
    text = text.strip()
    chunks, start = [], 0
    while start < len(text):
        piece = text[start:start + size].strip()
        if piece:
            chunks.append(piece)
        start += size - overlap
    return chunks


class VectorStore:
    def __init__(self):
        self.chunks = []      # list of (source, text)
        self.vectors = None   # numpy matrix

    def add_document(self, source: str, text: str) -> int:
        pieces = chunk_text(text)
        if not pieces:
            return 0
        vecs = _get_model().encode(pieces, normalize_embeddings=True)
        self.chunks += [(source, p) for p in pieces]
        self.vectors = vecs if self.vectors is None else np.vstack([self.vectors, vecs])
        return len(pieces)

    def search(self, query: str, k: int = 5):
        if self.vectors is None:
            return []
        q = _get_model().encode([query], normalize_embeddings=True)[0]
        scores = self.vectors @ q
        top = np.argsort(scores)[::-1][:k]
        return [(self.chunks[i][0], self.chunks[i][1], float(scores[i])) for i in top]
