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

    def generate_response(self, query, vector_store, history=None):
        # 1. Get the chunks most relevant to this question.
        retrieved_chunks = self.retrieval.retrieve(query, vector_store)

        # 2. Squash them into one block of text to feed the model.
        context = "\n".join([chunk.page_content for chunk in retrieved_chunks])

        # 3. Build the "conversation so far" text from the history list.
        #
        #    The LLM has NO memory of its own — every call is independent.
        #    "Chat memory" just means we paste the previous turns into the
        #    prompt so the model can resolve follow-ups like "what about
        #    its downsides?".
        #
        #    history looks like:
        #      [("user", "What is RAG?"),
        #       ("assistant", "RAG stands for..."),
        #       ("user", "What are its downsides?")]

        history_block = ""                    # stays empty on the first question

        if history:                          # None or [] -> skip this whole block
            lines = []                       # one formatted string per turn

            for turn in history:
                role = turn[0]               # "user" or "assistant"
                text = turn[1]               # what was said

                # "user" -> "User", "assistant" -> "Assistant" (cosmetic).
                role_label = role.capitalize()

                # e.g. "User: What is RAG?"
                one_line = role_label + ": " + text
                lines.append(one_line)

            # glue the turns together, one per line
            conversation = "\n".join(lines)

            # add a header and a blank line after it
            history_block = "Conversation so far:\n" + conversation + "\n\n"

        # 4. Build the final prompt: history, then document context, then
        # instructions, then the new question. The instruction to only use
        # the context is also what ragas_evals.py's "faithfulness" metric
        # checks — whether the answer actually stuck to this context.
        prompt = (
            history_block
            + "Context from the document:\n" + context + "\n\n"
            + "Answer the user's question using ONLY the context above. "
            + "If the answer is not in the context, say you don't know.\n\n"
            + "Question: " + query + "\nAnswer:"
        )

        response = self.model.invoke(prompt)
        return response.content
    
        