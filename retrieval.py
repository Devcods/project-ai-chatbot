# Step 4 of the pipeline: given a question, find the most relevant chunks.
from embedding import Embedding


class Retrieval:
    def __init__(self):
        # Note: this isn't actually used in retrieve() below. The vector
        # store already knows how to embed a query internally — it was
        # given an embeddings object when it was created in embedding.py —
        # so similarity_search() handles that on its own.
        self.embedding = Embedding()

    def retrieve(self, query, vector_store, k=5):
        # similarity_search embeds `query` the same way the chunks were
        # embedded, then returns the k chunks whose vectors are closest
        # (i.e. closest in MEANING, not exact word matches).
        results = vector_store.similarity_search(query, k=k)
        return results
    