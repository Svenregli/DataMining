import fitz  # PyMuPDF

def extract_text_from_pdf(file) -> str:
    with fitz.open(stream=file.read(), filetype="pdf") as doc:
        text = ""
        for page in doc:
            text += page.get_text()
    return text



# --- Chunk helper --- might be cleaner to from antoher file
def chunk_text(text, size=1000, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + size)
        chunks.append(text[start:end])
        start += size - overlap
        if start >= len(text):
            break
    if end < len(text) and start < len(text):
        chunks.append(text[start:])
    return chunks



