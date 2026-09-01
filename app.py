# ============================================================================
# app.py — the Streamlit web UI for the RAG PDF Q&A app
# ============================================================================
#
# HOW STREAMLIT WORKS (read this once, the rest of the file will make sense):
#
# 1. A Streamlit app is just a normal Python script. Streamlit runs it
#    top-to-bottom to draw the page.
#
# 2. Every time the user interacts with a widget (types in a box, clicks a
#    button, uploads a file), Streamlit RE-RUNS THE WHOLE SCRIPT again from
#    the top. There is no "onClick" callback like in JavaScript — the script
#    simply runs again, and this time the widget returns its new value.
#
# 3. Because the script re-runs constantly, anything slow or stateful needs
#    protecting:
#       - @st.cache_resource / @st.cache_data  -> "don't redo this every rerun"
#       - st.session_state                     -> "remember this between reruns"
#
# 4. Widgets are functions that RETURN their current value:
#       name = st.text_input("Your name")   # name is a str
#       go   = st.button("Go")              # go is True only on the rerun
#                                           # triggered by the click
# ============================================================================

import os
import tempfile

import streamlit as st

from llm import LLM
from chunk import Chunk
from embedding import Embedding


# ----------------------------------------------------------------------------
# PAGE CONFIG — must be the first Streamlit call in the script.
# Sets the browser tab title and how wide the content area is.
# ----------------------------------------------------------------------------
st.set_page_config(page_title="AI Document Q&A", page_icon="📄")


# ----------------------------------------------------------------------------
# CACHED HELPERS
#
# These two functions do the slow work (calling OpenAI). We wrap them in
# @st.cache_resource so Streamlit runs the body ONCE and then hands back the
# same object on every future rerun instead of recomputing.
#
# @st.cache_resource is for live objects/connections (a DB client, an ML
# model, a vector store). @st.cache_data is for plain data (a DataFrame,
# a dict). We have live objects here, so cache_resource.
# ----------------------------------------------------------------------------

@st.cache_resource(show_spinner="Reading and indexing your PDF (one-time)...")
def build_vector_store(file_bytes: bytes, file_name: str):
    """Turn an uploaded PDF into a searchable Chroma vector store.

    Streamlit decides "have I run this before?" by hashing the arguments.
    We pass `file_bytes` (raw bytes hash cleanly) instead of the upload
    object (which does not). Bonus: a different PDF -> different bytes ->
    cache miss -> it correctly rebuilds for the new file.
    """
    # PyPDFLoader needs a real file PATH on disk, not bytes in memory,
    # so write the upload to a temporary .pdf file first.
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        # Step 1+2: load the PDF and split it into overlapping text chunks.
        chunks = Chunk(tmp_path).create_chunks()

        # Step 3: embed every chunk with OpenAI and store the vectors in
        # Chroma. This is the expensive call the cache is protecting.
        vector_store = Embedding().create_vector_store(chunks)
        return vector_store
    finally:
        # Delete the temp file no matter what (even if the steps above fail).
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@st.cache_resource(show_spinner=False)
def get_llm() -> LLM:
    """Build the LLM client once and reuse it for every question."""
    return LLM()


# ----------------------------------------------------------------------------
# THE PAGE
# Everything below runs top-to-bottom on the first load and on every rerun.
# ----------------------------------------------------------------------------

st.title("📄 AI Document Q&A")
st.caption("Upload a PDF, ask a question, get an answer grounded in the document.")

# st.file_uploader returns None until a file is chosen, then an
# UploadedFile object. type="pdf" restricts what the user can pick.
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

# If nothing is uploaded yet, show a hint and stop this rerun here.
# st.stop() ends the script early so we don't run the Q&A code below
# with no document loaded.
if uploaded_file is None:
    st.info("Waiting for a PDF...")
    st.stop()

# .getvalue() returns the file's bytes without consuming the buffer,
# so it still works on later reruns. (.read() would empty it.)
file_bytes = uploaded_file.getvalue()

# First time for this file: slow (embeds the PDF). Every rerun after: instant.
vector_store = build_vector_store(file_bytes, uploaded_file.name)
st.success(f"Ready: **{uploaded_file.name}**")

# ----------------------------------------------------------------------------
# CHAT MEMORY
# st.session_state is a dict that survives reruns. We keep the whole
# conversation in st.session_state.messages so (a) the chat stays on screen
# and (b) we can feed past turns back to the LLM as memory.
# Each item is a dict: {"role": "user" | "assistant", "content": "..."}
# Always initialise a key before you read it.
# ----------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Re-draw the whole conversation on every rerun.
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):        # a "user" or "assistant" bubble
        st.write(msg["content"])

# st.chat_input is pinned to the bottom of the page. It returns the typed
# text once (on submit) and None on every other rerun.
question = st.chat_input("Ask a question about your document")

if question:
    # 1. Save and show the user's message.
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    # 2. Build the memory we pass to the LLM: the messages BEFORE this one,
    #    as (role, text) pairs. We keep only the last 6 (about 3 exchanges)
    #    so the prompt stays small and the OpenAI cost stays predictable.
    history = []
    for msg in st.session_state.messages[-7:-1]:
        history.append((msg["role"], msg["content"]))

    # 3. Ask the LLM, then show and save the answer.
    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking..."):
                answer = get_llm().generate_response(question, vector_store, history)
            st.write(answer)
        except Exception as e:
            # Show a friendly error instead of a raw traceback on the page.
            answer = f"Something went wrong: {e}"
            st.error(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
