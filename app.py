import os
import tempfile

import streamlit as st

from llm import LLM
from chunk import Chunk
from embedding import Embedding


# ─────────────────────────────────────────────────────────────
# CACHED RESOURCES
# These functions run ONCE and their return value is kept alive
# in memory across Streamlit reruns.
# ─────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Processing PDF — this runs only once per file...")
def build_vector_store(file_bytes: bytes, file_name: str):
    """
    Chunk the PDF and build the vector store.

    WHY @st.cache_resource:
      A vector store is a live in-memory object, not plain data.
      cache_resource keeps the SAME object alive across reruns
      instead of rebuilding (and re-paying OpenAI) every time.

    WHY file_bytes is the argument (and not the UploadedFile object):
      Streamlit decides "have I seen these arguments before?" by
      hashing them. Raw bytes hash cleanly. An UploadedFile object
      does not — Streamlit would either error or miss the cache.
      Passing bytes also means: upload a DIFFERENT pdf -> different
      bytes -> different hash -> cache miss -> correctly rebuilds.

    Args:
        file_bytes: raw contents of the uploaded PDF
        file_name:  original filename (part of the cache key)

    Returns:
        A vector store ready to be queried.
    """
    # Write the uploaded bytes to a real file on disk, because
    # PyPDFLoader (inside Chunk -> PdfLoader) needs a file PATH,
    # not an in-memory buffer.
    #
    # delete=False so the file survives after the `with` block closes;
    # we delete it ourselves in the finally block below.
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        # Step 1: load the PDF and split it into overlapping chunks
        chunker = Chunk(tmp_path)
        chunks = chunker.create_chunks()

        # Step 2: embed every chunk and index them in FAISS.
        # THIS is the expensive step — one OpenAI API call per batch
        # of chunks. Everything above exists to make sure it runs once.
        embedder = Embedding()
        vector_store = embedder.create_vector_store(chunks)

        return vector_store

    finally:
        # Always clean up the temp file, even if chunking/embedding
        # raised. Your old code left a stale temp.pdf in the repo root
        # and reused the same filename for every user and every upload.
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@st.cache_resource(show_spinner=False)
def get_llm():
    """
    Build the LLM client once.

    WHY cache this too:
      LLM() constructs a ChatOpenAI client (and, in your current code,
      a Retrieval -> Embedding -> OpenAIEmbeddings chain). Rebuilding
      that on every rerun is pure waste. No arguments means the cache
      key is constant, so this runs exactly once per session.
    """
    return LLM()


# ─────────────────────────────────────────────────────────────
# UI
# Everything below here re-runs top-to-bottom on every interaction.
# That is fine now, because the expensive work sits behind the
# cached functions above.
# ─────────────────────────────────────────────────────────────

def main():
    st.title("AI Document Q&A")

    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

    # Nothing uploaded yet — show a hint and stop.
    # st.stop() halts this rerun cleanly instead of nesting the whole
    # app inside an `if` block.
    if uploaded_file is None:
        st.info("Upload a PDF to get started.")
        st.stop()

    # Read the file into memory ONCE per rerun.
    # .getvalue() returns bytes and (unlike .read()) does not consume
    # the buffer, so it stays readable on later reruns.
    file_bytes = uploaded_file.getvalue()

    # First call for this file: builds and caches the store (slow).
    # Every later rerun with the same file: instant cache hit (free).
    vector_store = build_vector_store(file_bytes, uploaded_file.name)

    st.success(f"Ready: {uploaded_file.name}")

    question = st.text_input("Ask a question about your document")

    if st.button("Get Answer"):
        # Guard against an empty submission
        if not question.strip():
            st.warning("Please enter a question first.")
            st.stop()

        llm = get_llm()

        # Wrap the API call. A network blip or a bad key should show
        # the user an error, not dump a raw traceback into the page.
        try:
            with st.spinner("Thinking..."):
                answer = llm.generate_response(question, vector_store)
            st.write(answer)
        except Exception as e:
            st.error(f"Something went wrong generating the answer: {e}")


if __name__ == "__main__":
    main()