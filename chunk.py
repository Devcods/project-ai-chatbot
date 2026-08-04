# Step 2 of the pipeline: split full PDF pages into smaller pieces.
#
# WHY split at all: embedding models and the LLM's context window both have
# limits, and retrieval works better over small, focused pieces of text than
# over whole pages — a question usually only needs one paragraph's worth of
# context, not an entire page.
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pdf_loader import PdfLoader


class Chunk:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def create_chunks(self):
        pdf_loader = PdfLoader(self.file_path)
        documents = pdf_loader.load()

        # chunk_size=1000: max characters per chunk.
        # chunk_overlap=200: each chunk repeats the last 200 characters of
        # the previous one, so a sentence that gets cut at a chunk boundary
        # still appears in full in at least one chunk.
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)

        return chunks
    