from langchain_text_splitters import RecursiveCharacterTextSplitter
from pdf_loader import PdfLoader

class Chunk:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def create_chunks(self):
        pdf_loader = PdfLoader(self.file_path)
        documents = pdf_loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)
        
        return chunks
    