# AI Document QA

This is a simple RAG based PDF question answering app.

The user can upload a PDF, ask a question, and get an answer based on the PDF content. The project uses LangChain, OpenAI, FAISS, and Streamlit. I also added RAGAS evaluation to check the quality of the answers.

# Main Features

Upload a PDF.

Split the PDF into smaller text chunks.

Create embeddings from the chunks.

Store the embeddings in FAISS.

Retrieve the most relevant chunks for a question.

Generate an answer using an OpenAI model.

Evaluate the system using RAGAS.

# Main Files

app.py handles the Streamlit user interface.

pdf_loader.py loads the PDF.

chunk.py splits the PDF text.

embedding.py creates the FAISS vector store.

retrieval.py finds relevant chunks.

llm.py generates answers using the retrieved context.

evaluate.py runs RAGAS evaluation.

test_set.json stores sample evaluation questions.

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

    python evaluate.py

The results are printed in the terminal and saved in evaluation_results.csv.

# Future Improvements

Save the FAISS vector store so the PDF does not need to be embedded again every time.

Support multiple PDFs.

Improve the UI.

Add streaming answers.

Create a larger test set for better evaluation.