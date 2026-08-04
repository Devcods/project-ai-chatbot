# Step 3 of the pipeline: turn text chunks into vectors and make them
# searchable.
#
# An "embedding" is just a list of numbers that represents the MEANING of a
# piece of text. Chunks about similar topics end up with similar numbers,
# which is what lets us later search by meaning instead of exact keywords.
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
import os
load_dotenv()


class Embedding:
    def __init__(self):
        # Calls OpenAI's embedding model. This same object is also handed to
        # Chroma below, so Chroma knows how to embed future SEARCH QUERIES
        # the same way it embedded these chunks — otherwise the vectors
        # wouldn't be comparable to each other.
        self.embeddings = OpenAIEmbeddings()

    def create_vector_store(self, chunks):
        # Chroma.from_documents embeds every chunk (one OpenAI API call) and
        # stores the resulting vectors in memory so they can be searched by
        # similarity later. No persist_directory is passed, so nothing is
        # written to disk — the store lives only as long as this Python
        # process (matches app.py rebuilding it fresh per uploaded file).
        vector_store = Chroma.from_documents(chunks, self.embeddings)
        return vector_store
 
 