import os
from pathlib import Path
from typing import TypedDict, List, Optional
from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import HumanMessage
from app.graph.state import AppState

# Initial LLM setup
llm = ChatOllama(model="llama3")

SYSTEM_PROMPT = """
You are a helpful assistant that can:
- answer user questions
- route user requests to internal tools if they mention video transcripts, summarization, redaction, OCR, or RAG.

Detect the intent of the user based on their message and set one of these task intents:
- "youtube_transcript"
- "summarize_doc"
- "redact_doc"
- "ocr_image"
- "rag_ingest"

If no such task is requested, respond normally and leave task_intent empty.
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="messages"),
    ("human", "{input}")
])

chain = prompt | llm


def chat_agent_node(state: AppState) -> AppState:
    messages = state.get("messages", [])
    latest_user_msg = messages[-1]["content"] if messages else ""
    history = [HumanMessage(content=m["content"]) for m in messages]

    # LLM response
    result = chain.invoke({"messages": history, "input": latest_user_msg})
    assistant_response = result.content

    state.setdefault("messages", [])
    state["messages"].append({
        "role": "assistant",
        "content": assistant_response
    })

    # Rudimentary intent detection
    if "transcript" in latest_user_msg.lower():
        state["task_intent"] = "youtube_transcript"
    elif "summarize" in latest_user_msg.lower():
        state["task_intent"] = "summarize_doc"
    elif "redact" in latest_user_msg.lower():
        state["task_intent"] = "redact_doc"
    elif "ocr" in latest_user_msg.lower() or "image text" in latest_user_msg.lower():
        state["task_intent"] = "ocr_image"
    elif "rag" in latest_user_msg.lower() or "ingest" in latest_user_msg.lower():
        state["task_intent"] = "rag_ingest"
    else:
        state["task_intent"] = ""

    return state
