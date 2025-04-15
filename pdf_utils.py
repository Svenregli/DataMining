import fitz  # PyMuPDF

def extract_text_from_pdf(file) -> str:
    with fitz.open(stream=file.read(), filetype="pdf") as doc:
        text = ""
        for page in doc:
            text += page.get_text()
    return text
