from langgraph.graph import StateGraph, END
from app.graph.state import AppState
from app.agents.chat_agent import chat_agent_node
from app.agents.transcript_agent import transcript_agent_node  
from app.agents.rag_agent import rag_agent_node
from app.agents.summary_agent import doc_summary_agent_node
from app.agents.redactor_agent import doc_redactor_agent_node
from app.agents.ocr_agent import ocr_agent_node



def build_graph():
    g = StateGraph(AppState)

    # Entry node
    g.add_node("chat_agent", chat_agent_node)
    g.add_node("transcript_agent", transcript_agent_node)  
    g.add_node("rag_agent", rag_agent_node)
    g.add_node("summary_agent", doc_summary_agent_node)
    g.add_node("redactor_agent", doc_redactor_agent_node)
    g.add_node("ocr_agent", ocr_agent_node)
    g.set_entry_point("chat_agent")

    # For now, we’ll just end after chat (we’ll add branches next)
    g.add_conditional_edges(
        "chat_agent",
        lambda state: state.get("task_intent", ""),
        {
            "": END,
            "youtube_transcript": END,  
            "summarize_doc": END,
            "redact_doc": END,
            "ocr_image": END,
            "rag_ingest": END
        }
    )

    # Add edges between nodes
    g.add_edge("transcript_agent", "chat_agent")
    g.add_edge("rag_agent", "chat_agent")
    g.add_edge("summary_agent", "chat_agent")
    g.add_edge("redactor_agent", "chat_agent")
    g.add_edge("ocr_agent", "chat_agent")

    return g.compile()
