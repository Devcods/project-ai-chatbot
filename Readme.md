# AI Document QA

This is a simple RAG based PDF question answering app.

The user can upload a PDF and have a conversation about its content — ask a question, get an answer grounded in the PDF, then ask follow-up questions that refer back to earlier answers. The project uses LangChain, OpenAI, Chroma, and Streamlit. I also added RAGAS evaluation to check the quality of the answers.

# Main Features

Upload a PDF.

Split the PDF into smaller text chunks.

Create embeddings from the chunks.

Store the embeddings in Chroma.

Retrieve the most relevant chunks for a question.

Generate an answer using an OpenAI model.

Remember the conversation so far and feed recent turns back to the model, so follow-up questions ("explain that", "the first one") work.

Evaluate the system using RAGAS.

# Main Files

app.py handles the Streamlit chat user interface and keeps the conversation history in session state.

pdf_loader.py loads the PDF.

chunk.py splits the PDF text.

embedding.py creates the Chroma vector store.

retrieval.py finds relevant chunks.

llm.py generates answers using the retrieved context plus the recent chat history.

ragas_evals.py runs RAGAS evaluation.

test_set.json stores sample evaluation questions.

# How the Pipeline Works

When the app starts, it runs an ingestion pipeline once per uploaded PDF:

1. Load. app.py writes the uploaded PDF to a temporary file, then pdf_loader.py uses PyPDFLoader to read it into one text document per page.

2. Chunk. chunk.py splits those pages with RecursiveCharacterTextSplitter into pieces of about 1000 characters with 200 characters of overlap, so a sentence cut at a boundary still appears whole in one chunk.

3. Embed and store. embedding.py sends every chunk to OpenAI's embedding model and stores the resulting vectors in an in-memory Chroma vector store. This is the slow, paid step, so app.py caches the result with st.cache_resource and only rebuilds it when a different PDF is uploaded.

Then, for every question the user asks, it runs the query pipeline:

4. Retrieve. retrieval.py embeds the question the same way and asks Chroma for the 5 chunks whose vectors are closest in meaning.

5. Build the prompt. llm.py joins those chunks into a context block, prepends the recent chat history (the last few user and assistant turns, kept in st.session_state by app.py), and adds the new question. History lets the model resolve follow-ups; the context block is the document text it must answer from.

6. Generate. llm.py sends that prompt to gpt-3.5-turbo and returns the answer, which app.py shows in the chat and saves to the history for the next turn.

The retrieved chunks are used for one question only and then discarded; retrieval runs fresh for the next question. Only the user questions and the model answers are kept as history.

# How to Run

Install the required libraries.

    pip install -r requirements.txt

Add your OpenAI API key in a .env file.

    OPENAI_API_KEY=your_api_key_here

Run the app.

    streamlit run app.py

# Evaluation

To run the RAGAS evaluation, update test_set.json with test questions and answers.

Then run:

    python ragas_evals.py

The results are printed in the terminal and saved in evaluation_results.csv.

# Future Improvements

Persist the Chroma vector store to disk so the PDF does not need to be embedded again every time.

Support multiple PDFs.

Improve the UI.

Add streaming answers.

Make retrieval history-aware: rewrite a follow-up question into a standalone query before searching, so retrieval works as well as the answering does.

Create a larger test set for better evaluation.