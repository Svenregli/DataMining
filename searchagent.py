from __future__ import annotations as _annotations

import os
from dataclasses import dataclass
from datetime import datetime
from dotenv import load_dotenv
from httpx import AsyncClient
from openai import AsyncOpenAI
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from tools import extract_pdf_text_tool  # Import the tool

load_dotenv()
llm = os.getenv("LLM_MODEL", "gpt-4o")

if llm.lower().startswith("gpt"):
    model = OpenAIModel(llm)
else:
    model = OpenAIModel(llm, openai_client=AsyncOpenAI(base_url='http://localhost:11434/v1', api_key='ollama'))

@dataclass
class Deps:
    client: AsyncClient
    brave_api_key: str | None = None

# Existing topic agent
web_search_agent = Agent(
    model,
    system_prompt="You are an expert in summarizing academic papers.",
    deps_type=Deps,
    retries=2,
)

# 🧠 Tool-enabled agent
pdf_tool_agent = Agent(
    model,
    system_prompt="You are an academic assistant that analyzes PDF papers. Use the available tool to extract PDF text from a URL before answering.",
    tools=[extract_pdf_text_tool],
    retries=2
)


#  Agent to extract dependent and independent variables
variable_extraction_agent = Agent(
    model,
    system_prompt=(
        "You are an academic assistant that extracts variables from research papers. "
        "Use the provided tool to read a paper from a PDF URL. "
        "Then identify the dependent and independent variables in the study. "
        "Respond clearly, listing each variable."
    ),
    tools=[extract_pdf_text_tool],
    retries=2
)

__all__ = ["web_search_agent", "pdf_tool_agent", "variable_extraction_agent", "reference_extraction_agent", "Deps"]
