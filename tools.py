import fitz as pymupdf  # PyMuPDF
import requests
from pydantic import BaseModel
from typing import Annotated
from pydantic_ai.tools import tool


class PDFInput(BaseModel):
    pdf_url: Annotated[str, "URL to the PDF file to extract"]


@tool(name="extract_pdf_text_tool", description="Download a PDF from a URL and return its text.")
def extract_pdf_text_tool(input: PDFInput) -> str:
    try:
        response = requests.get(input.pdf_url)
        response.raise_for_status()

        with open("temp.pdf", "wb") as f:
            f.write(response.content)

        text = ""
        with pymupdf.open("temp.pdf") as doc:
            for page in doc:
                text += page.get_text()

        return text[:4000]  # Limit to first 4000 characters (adjust as needed)
    except Exception as e:
        return f"Failed to extract PDF: {e}"
