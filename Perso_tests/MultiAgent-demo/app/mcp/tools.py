# app/mcp/tools.py

from langgraph.mcp.tool import tool
from app.tools.youtube_transcript import get_transcript
from app.tools.ocr_utils import extract_text_from_image
from app.tools.rag_utils import ingest_doc, query_doc
from app.tools.redactor_utils import redact_document
from app.tools.summary_utils import summarize_doc

@tool(name="youtube_transcript", description="Extract transcript from YouTube video URL")
def youtube_transcript_tool(input: dict) -> dict:
    return {"transcript": get_transcript(input["url"])}

@tool(name="ocr_image", description="Extract text from image file path")
def ocr_image_tool(input: dict) -> dict:
    return {"text": extract_text_from_image(input["path"])}

@tool(name="summarize_doc", description="Summarize uploaded DOCX or PDF file")
def summarize_doc_tool(input: dict) -> dict:
    return {"summary": summarize_doc(input["path"])}

@tool(name="redact_doc", description="Redact PPI from a document")
def redact_doc_tool(input: dict) -> dict:
    return {"redacted_path": redact_document(input["path"])}
