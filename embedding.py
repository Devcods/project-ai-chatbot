from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import os
load_dotenv()

class Embedding:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
       
    def create_vector_store(self, chunks):
        vector_store = FAISS.from_documents(chunks, self.embeddings)
        return vector_store
 
 