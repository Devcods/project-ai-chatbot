# AI Document Q&A

A Retrieval-Augmented Generation (RAG) chatbot that lets you upload a PDF and ask questions about it. Built with LangChain, OpenAI, FAISS, and Streamlit, with RAGAS-powered evaluation.

---

## Architecture

```
PDF Upload → PdfLoader → Chunk → Embedding (FAISS) → Retrieval → LLM → Answer
                                                                     ↑
                                                              evaluate.py (RAGAS)
```

| File | Responsibility |
|---|---|
| `app.py` | Streamlit UI — file upload, question input, answer display |
| `pdf_loader.py` | Loads a PDF into LangChain `Document` objects via PyPDF |
| `chunk.py` | Splits documents into overlapping chunks (1 000 chars / 200 overlap) |
| `embedding.py` | Generates OpenAI embeddings and builds a FAISS vector store |
| `retrieval.py` | Runs similarity search against the vector store (top-k chunks) |
| `llm.py` | Calls GPT-3.5-Turbo with retrieved context to generate an answer |
| `evaluate.py` | RAGAS evaluation suite (faithfulness, relevancy, precision, recall) |

---

## Quick Start

### 1. Clone and install dependencies

```bash
git clone <your-repo-url>
cd <repo-directory>
pip install -r requirements.txt
pip install ragas datasets  # evaluation extras
```

### 2. Set up your OpenAI API key

```bash
cp .env.example .env
# then edit .env and add your key:
# OPENAI_API_KEY=sk-...
```

### 3. Run the app

```bash
streamlit run app.py
```

Upload a PDF, type a question, and click **Get Answer**.

---

## RAGAS Evaluation

[RAGAS](https://docs.ragas.io) measures four aspects of RAG quality:

| Metric | What it measures | Needs ground truth? |
|---|---|---|
| **Faithfulness** | Is the answer grounded in the retrieved context? | No |
| **Answer Relevancy** | Is the answer relevant to the question? | No |
| **Context Precision** | Are the retrieved chunks actually useful? | No |
| **Context Recall** | Did retrieval capture all necessary information? | Yes |

### Option A — Evaluate with pre-collected data (fastest)

Edit `test_set.json` with your questions, ground-truth answers, and optionally pre-fetched answers/contexts, then run:

```bash
python evaluate.py
```

Results are printed to the console and saved to `evaluation_results.csv`.

### Option B — Evaluate by running the full pipeline live

Use `evaluate_from_pipeline()` to automatically retrieve chunks and generate answers during evaluation:

```python
from chunk import Chunk
from embedding import Embedding
from evaluate import RAGASEvaluator

# Build vector store from your PDF
chunks = Chunk("your_document.pdf").create_chunks()
vector_store = Embedding().create_vector_store(chunks)

evaluator = RAGASEvaluator()
questions, _, _, ground_truths = evaluator.load_test_set("test_set.json")

results = evaluator.evaluate_from_pipeline(questions, ground_truths, vector_store)
evaluator.save_results(results, "evaluation_results.csv")
```

### Building a test set

Fill in `test_set.json` following this schema:

```json
[
  {
    "question":     "What does the document say about X?",
    "ground_truth": "The document states that X is ...",
    "answer":       "(optional) leave empty to run the pipeline live",
    "contexts":     ["(optional) pre-fetched chunk 1", "chunk 2"]
  }
]
```

Tips for writing good test questions:
- Cover key facts, comparisons, and summaries from your document.
- Write `ground_truth` answers as complete sentences based on what the document actually says.
- Aim for at least 10–20 questions per document for meaningful scores.

---

## Project Structure

```
.
├── app.py                  # Streamlit application
├── pdf_loader.py           # PDF ingestion
├── chunk.py                # Text chunking
├── embedding.py            # FAISS vector store
├── retrieval.py            # Similarity search
├── llm.py                  # LLM response generation
├── evaluate.py             # RAGAS evaluation
├── test_set.json           # Sample evaluation test set
├── requirements.txt        # Core dependencies
└── .env                    # API keys (not committed)
```

---

## Configuration

| Parameter | Location | Default | Notes |
|---|---|---|---|
| Chunk size | `chunk.py` | 1 000 chars | Increase for longer context windows |
| Chunk overlap | `chunk.py` | 200 chars | Higher overlap reduces missed boundaries |
| Retrieved chunks (k) | `retrieval.py` | 5 | Raise k for broader context; watch token limits |
| LLM model | `llm.py` | `gpt-3.5-turbo` | Swap for `gpt-4o` etc. |
| LLM temperature | `llm.py` | 0.7 | Lower for more deterministic answers |
| Eval model | `evaluate.py` | `gpt-3.5-turbo` | Used by RAGAS metrics |

---

## Requirements

```
langchain
langchain-community
langchain-openai
openai
faiss-cpu
streamlit
pypdf
python-dotenv
ragas
datasets
```

---

## Roadmap

- [ ] Persist the vector store to disk (avoid re-embedding on reload)
- [ ] Support multiple PDFs in one session
- [ ] Add streaming responses
- [ ] Automated CI evaluation against a fixed test set
- [ ] Swap FAISS for a managed vector DB (Pinecone, Weaviate, etc.)