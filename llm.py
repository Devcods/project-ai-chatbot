# Step 5 (final step) of the pipeline: retrieve context, then ask the LLM
# to answer using it. This is the "Generation" half of Retrieval-Augmented
# Generation (RAG) — retrieval.py is the "Retrieval" half.
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
from retrieval import Retrieval


class LLM:
    def __init__(self):
        load_dotenv()
        self.apikey = os.getenv("OPENAI_API_KEY")

        # temperature=0.7 allows some variation in wording, which is fine
        # for a user-facing answer. Compare this to ragas_evals.py, which
        # uses temperature=0 for its judge model — there, consistent
        # grading matters more than natural-sounding text.
        self.model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7, openai_api_key=self.apikey)
        self.retrieval = Retrieval()

    def generate_response(self, query, vector_store):
        # 1. Get the chunks most relevant to this question.
        retrieved_chunks = self.retrieval.retrieve(query, vector_store)

        # 2. Squash them into one block of text to feed the model.
        context = "\n".join([chunk.page_content for chunk in retrieved_chunks])

        # 3. Build a prompt that tells the model: only answer using this
        # context, don't just answer from what it already knows. (This is
        # also exactly what ragas_evals.py's "faithfulness" metric checks —
        # whether the answer actually stuck to this context.)
        prompt = f"Context: {context}\n\nQuestion: {query}\nAnswer:"

        response = self.model.invoke(prompt)
        return response.content
    
        