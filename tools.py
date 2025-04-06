# tools.py
import fitz  # PyMuPDF
import requests

def extract_pdf_text_tool(pdf_url: str) -> str:
    try:
        response = requests.get(pdf_url)
        response.raise_for_status()

        with open("temp.pdf", "wb") as f:
            f.write(response.content)

        text = ""
        with fitz.open("temp.pdf") as doc:
            for page in doc:
                text += page.get_text()

        return text[:4000]
    except Exception as e:
        return f"Failed to extract PDF: {e}"
