from embedding import Embedding

class Retrieval:
    def __init__(self):
        self.embedding = Embedding()
    
    def retrieve(self, query, vector_store, k=5):
        results = vector_store.similarity_search(query, k=k)
        return results
    