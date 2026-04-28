import streamlit as st
from pypdf import PdfReader
from langchain_groq import ChatGroq
import requests
import json
import os
import re
from dotenv import load_dotenv

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    st.error("Missing GROQ_API_KEY. Add it to your .env file and restart Streamlit.")
    st.stop()

model = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
    max_tokens=4000,
    timeout=30,
    api_key=groq_api_key
)

def extract_references_section(full_text: str) -> str:
    """Pull out just the references section from the paper."""
    # References usually start with these headers
    markers = ["References", "REFERENCES", "Bibliography", "BIBLIOGRAPHY"]
    
    for marker in markers:
        if marker in full_text:
            return full_text[full_text.index(marker):]
    
    return full_text[int(len(full_text) * 0.8):]

def split_text_into_chunks(text: str, chunk_size: int = 5000, overlap: int = 300) -> list[str]:
    """Split long text into overlapping chunks to keep model requests small."""
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunks.append(text[start:end])
        if end == text_length:
            break
        start = end - overlap

    return chunks


def extract_citations(text: str) -> str:
    """Extract all references from research paper text. Return as JSON list with title, authors, year fields."""
    response = model.invoke(f"""
    Extract all references from the text below.
    Return ONLY a JSON list, no explanation, no markdown backticks.
    Each item must have: title, authors, year.
    
    Text: {text}
    """)
    return response.content


def parse_citations_response(raw_response: str) -> list[dict]:
    """Parse model response into a citation list, with a fallback for markdown code blocks."""
    try:
        parsed = json.loads(raw_response)
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        pass

    json_block_match = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", raw_response, re.DOTALL)
    if json_block_match:
        try:
            parsed = json.loads(json_block_match.group(1))
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            return []

    return []


def verify_citation(title: str) -> str:
    """Search Semantic Scholar to verify a citation exists by title."""
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {"query": title, "limit": 1, "fields": "title,authors,year"}
    response = requests.get(url, params=params)
    data = response.json()
    
    if data.get("total", 0) > 0:
        match = data["data"][0]
        return f"VERIFIED: {match['title']} ({match.get('year', 'unknown')})"
    return f"NOT FOUND: {title}"


def extract_citations_in_chunks(references_text: str) -> list[dict]:
    """Extract citations from multiple chunks and deduplicate by title."""
    chunks = split_text_into_chunks(references_text)
    all_citations = []

    for chunk in chunks:
        raw = extract_citations(chunk)
        citations = parse_citations_response(raw)
        all_citations.extend(citations)

    deduped = {}
    for citation in all_citations:
        if not isinstance(citation, dict):
            continue
        title = str(citation.get("title", "")).strip()
        if not title:
            continue
        normalized = title.casefold()
        if normalized not in deduped:
            deduped[normalized] = {
                "title": title,
                "authors": citation.get("authors", ""),
                "year": citation.get("year", ""),
            }

    return list(deduped.values())


def build_verification_report(citations: list[dict]) -> str:
    """Verify extracted citations and produce a human-readable summary."""
    if not citations:
        return "No citations could be extracted from the references section."

    verified = []
    not_found = []

    for citation in citations:
        title = str(citation.get("title", "")).strip()
        if not title:
            continue

        verification = verify_citation(title)
        if verification.startswith("VERIFIED"):
            verified.append(verification)
        else:
            not_found.append(verification)

    lines = [
        f"Total extracted citations: {len(citations)}",
        f"Verified: {len(verified)}",
        f"Not found: {len(not_found)}",
        "",
        "Verified citations:",
    ]
    lines.extend(verified if verified else ["- None"])
    lines.append("")
    lines.append("Not found citations:")
    lines.extend(not_found if not_found else ["- None"])

    return "\n".join(lines)


st.title("Scila - Scientific Paper Analysis")

paper = st.file_uploader("Upload your paper", type=["pdf"])

if paper is None:
        st.write("Please upload a paper to proceed.")
else:
    st.write("Paper uploaded successfully!")

    paperPdf = PdfReader(paper)
    full_text = ""

    for page in paperPdf.pages:
        full_text += page.extract_text() + "\n"

    if st.button("Verify Citations"):
        with st.spinner("Analyzing paper..."):
            references_text = extract_references_section(full_text)
            citations = extract_citations_in_chunks(references_text)
            report = build_verification_report(citations)
            st.text(report)

