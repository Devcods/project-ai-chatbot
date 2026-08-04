# Step 1 of the pipeline: turn a PDF file on disk into text LangChain can work with.
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader


class PdfLoader:
    """Wraps PyPDFLoader so the rest of the app doesn't need to know
    which PDF library is being used underneath — just call .load()."""

    def __init__(self, file_path: str):
        # file_path must be a real path on disk, not raw bytes — that's why
        # app.py writes the uploaded file to a temp file before calling this.
        self.file_path = file_path

    def load(self):
        loader = PyPDFLoader(self.file_path)

        # documents is a list of LangChain Document objects, ONE PER PAGE.
        # Each has .page_content (the page's text) and .metadata (e.g. page
        # number). Chunk.create_chunks() later splits these pages further
        # into smaller pieces.
        documents = loader.load()
        return documents



