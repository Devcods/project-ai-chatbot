from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
from retrieval import Retrieval

class LLM:
    def __init__(self):
        load_dotenv()
        self.apikey = os.getenv("OPENAI_API_KEY")
        self.model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7, openai_api_key=self.apikey)
        self.retrieval = Retrieval()
    def generate_response(self, query, vector_store):
        retrieved_chunks = self.retrieval.retrieve(query, vector_store)
        context = "\n".join([chunk.page_content for chunk in retrieved_chunks])
        prompt = f"Context: {context}\n\nQuestion: {query}\nAnswer:"
        response = self.model.invoke(prompt)
        return response.content
    
        