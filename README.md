# Simple RAG App (3 free LLMs)

Upload PDF / CSV / HTML / TXT / MD / DOCX -> ask questions -> get answers
from Groq, Google Gemini and OpenRouter side by side.

## How it works
1. Upload -> text is extracted (`rag/loaders.py`)
2. Text is split into ~1000-char chunks and embedded locally with
   `all-MiniLM-L6-v2` (free, no API key) (`rag/store.py`)
3. Your question is embedded, top-k similar chunks are found
4. Chunks + question go to 3 LLMs in parallel (`rag/llms.py`)
5. Streamlit shows the 3 answers (`app.py`)

## Setup
1. Python 3.10+ required.
2. `python -m venv venv` then activate it
   - Windows: `venv\Scripts\activate`
   - Mac/Linux: `source venv/bin/activate`
3. `pip install -r requirements.txt`
4. Get free API keys:
   - Groq: https://console.groq.com/keys
   - Gemini: https://aistudio.google.com/apikey
   - OpenRouter: https://openrouter.ai/keys
5. Copy `.env.example` to `.env` and paste the keys
   (you can use just 1 or 2 keys; models without a key show an error message).
6. Run: `streamlit run app.py`

## Troubleshooting
- Model error 404/400: the provider renamed or retired a model. Change the
  model name in `.env` (see each provider's model list).
- OpenRouter `:free` models can be busy or rate-limited; try another free model.
- First run is slow: it downloads the embedding model (~90 MB) once.
- Old `.doc` files: save as `.docx` first.
- Scanned PDFs (images only) have no text; they need OCR.

## Ideas to extend
- Persist vectors (FAISS / Chroma) so you don't re-process files
- Chat history, source citations, OCR for scanned PDFs
