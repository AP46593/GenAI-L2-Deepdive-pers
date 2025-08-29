# app/mcp/registry.py

from langgraph.mcp.registry import ToolRegistry
from app.mcp.tools import (
    youtube_transcript_tool,
    ocr_image_tool,
    summarize_doc_tool,
    redact_doc_tool
)

# Register your tools
registry = ToolRegistry(tools=[
    youtube_transcript_tool,
    ocr_image_tool,
    summarize_doc_tool,
    redact_doc_tool
])
