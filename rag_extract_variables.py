import openai
from dotenv import load_dotenv
import os
import time
import streamlit as st
from embed_papers_openai import search_chunks

# Load env variables
load_dotenv()
openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def extract_variables_from_chunks(chunks: list[str]) -> str:
    if not chunks:
        return "No text chunks provided for variable extraction."

    context = "\n\n".join(chunks)

    prompt = f"""You are an academic assistant.

Given these excerpts from a research paper, identify the independent and dependent variables mentioned. If no variables are clearly mentioned, state that.

### Excerpts:
{context}

### Please respond ONLY in this format:
Independent Variables: [List variables or state \"None mentioned\"]
Dependent Variables: [List variables or state \"None mentioned\"]
"""
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful academic assistant specialized in identifying research variables."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=300
        )
        return response.choices[0].message.content.strip()

    except openai.RateLimitError as e:
        print(f"OpenAI Rate Limit Error: {e}")
        return "Error: API rate limit exceeded. Please try again later."
    except openai.InvalidRequestError as e:
        print(f"OpenAI Invalid Request Error: {e}")
        if "maximum context length" in str(e):
             return f"Error: The provided text is too long for the model ({e}). Consider processing fewer chunks."
        return f"Error: Invalid request to OpenAI API ({e})."
    except openai.APIError as e:
        print(f"OpenAI API Error: {e}")
        return "Error: An issue occurred with the OpenAI API. Please try again."
    except Exception as e:
        print(f"An unexpected error occurred during variable extraction: {e}")
        return "Error: An unexpected issue occurred."


def summarize_chunks(chunks: list[str]) -> str:
    if not chunks:
        return "No text chunks provided for summarization."

    context = "\n\n".join(chunks)

    prompt = f"""You are a research assistant. Read the following excerpts from a research paper and provide a concise summary covering the key findings, methods, and conclusions mentioned in the text. Present the summary in clear bullet points.

### Excerpts:
{context}

### Concise Summary (Bullet Points):"""

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful research assistant skilled at summarizing academic texts."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=500
        )
        return response.choices[0].message.content.strip()

    except openai.RateLimitError as e:
        print(f"OpenAI Rate Limit Error: {e}")
        return "Error: API rate limit exceeded. Please try again later."
    except openai.InvalidRequestError as e:
        print(f"OpenAI Invalid Request Error: {e}")
        if "maximum context length" in str(e):
             return f"Error: The provided text is too long for the model ({e}). Consider processing fewer chunks or using iterative summarization."
        return f"Error: Invalid request to OpenAI API ({e})."
    except openai.APIError as e:
        print(f"OpenAI API Error: {e}")
        return "Error: An issue occurred with the OpenAI API. Please try again."
    except Exception as e:
        print(f"An unexpected error occurred during summarization: {e}")
        return "Error: An unexpected issue occurred."
