import streamlit as st
from llm import LLM
from chunk import Chunk
from embedding import Embedding

def main():
    st.title("AI Document Q&A")
    
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
    
    if uploaded_file is not None:
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        chunker = Chunk("temp.pdf")
        chunks = chunker.create_chunks()
        
        embedder = Embedding()
        vector_store = embedder.create_vector_store(chunks)
        
        question = st.text_input("Ask a question about your document")
        
        if st.button("Get Answer"):
            llm = LLM()
            answer = llm.generate_response(question, vector_store)
            st.write(answer)

if __name__ == "__main__":
    main()

