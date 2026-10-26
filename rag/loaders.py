"""Read text out of uploaded files: pdf, csv, html, txt/md, docx."""
import io
import os


def load_file(filename: str, data: bytes) -> str:
    ext = os.path.splitext(filename)[1].lower()

    if ext == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(data))
        return "\n".join((p.extract_text() or "") for p in reader.pages)

    if ext == ".csv":
        import pandas as pd
        df = pd.read_csv(io.BytesIO(data))
        # one line per row so each row stays meaningful after chunking
        lines = []
        for _, row in df.iterrows():
            lines.append("; ".join(f"{c}={row[c]}" for c in df.columns))
        return "\n".join(lines)

    if ext in (".html", ".htm"):
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(data.decode("utf-8", errors="ignore"), "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        return soup.get_text(separator="\n")

    if ext == ".docx":
        import docx
        d = docx.Document(io.BytesIO(data))
        return "\n".join(p.text for p in d.paragraphs)

    if ext in (".txt", ".md", ".json", ".log"):
        return data.decode("utf-8", errors="ignore")

    raise ValueError(f"Unsupported file type: {ext} (old .doc? save it as .docx)")
