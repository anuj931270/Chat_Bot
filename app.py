import streamlit as st
from dotenv import load_dotenv

from rag.loaders import load_file
from rag.store import VectorStore
from rag.llms import build_prompt, ask_all

load_dotenv()
st.set_page_config(page_title="Simple RAG", page_icon="📚", layout="wide")
st.title("📚 Simple RAG – ask your documents, 3 LLMs answer")

if "store" not in st.session_state:
    st.session_state.store = VectorStore()
    st.session_state.done = set()

with st.sidebar:
    st.header("1. Upload documents")
    files = st.file_uploader(
        "PDF, CSV, HTML, TXT, MD, DOCX",
        type=["pdf", "csv", "html", "htm", "txt", "md", "docx"],
        accept_multiple_files=True,
    )
    if st.button("Process documents", type="primary") and files:
        for f in files:
            if f.name in st.session_state.done:
                continue
            with st.spinner(f"Reading {f.name}..."):
                try:
                    text = load_file(f.name, f.getvalue())
                    n = st.session_state.store.add_document(f.name, text)
                    st.session_state.done.add(f.name)
                    st.success(f"{f.name}: {n} chunks")
                except Exception as e:
                    st.error(f"{f.name}: {e}")
    if st.button("Clear all"):
        st.session_state.store = VectorStore()
        st.session_state.done = set()
        st.rerun()
    st.caption(f"Loaded: {', '.join(st.session_state.done) or 'nothing yet'}")
    top_k = st.slider("Chunks to retrieve", 2, 10, 5)

st.header("2. Ask a question")
q = st.text_input("Your question about the uploaded documents")

if q:
    if not st.session_state.done:
        st.warning("Upload and process a document first.")
    else:
        hits = st.session_state.store.search(q, k=top_k)
        with st.spinner("Asking 3 models..."):
            answers = ask_all(build_prompt(q, hits))
        cols = st.columns(3)
        for col, (name, ans) in zip(cols, answers.items()):
            with col:
                st.subheader(name)
                st.write(ans)
        with st.expander("Retrieved context"):
            for src, text, score in hits:
                st.markdown(f"**{src}** (score {score:.2f})")
                st.text(text[:600])
