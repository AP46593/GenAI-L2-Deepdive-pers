from pathlib import Path
from app.graph.state import AppState
from app.tools.redact_utils import redact_docx, redact_pdf

def doc_redactor_agent_node(state: AppState) -> AppState:
    files = state.get("uploaded_files", [])
    state["messages"].append({"role": "assistant", "content": "🔍 Redacting personal info from uploaded files..."})

    redacted_file_path = None

    for file_path in files:
        ext = Path(file_path).suffix.lower()
        if ext == ".docx":
            redacted_file_path = redact_docx(Path(file_path))
        elif ext == ".pdf":
            redacted_file_path = redact_pdf(Path(file_path))
        else:
            continue

        if redacted_file_path:
            state["redacted_file"] = str(redacted_file_path)
            state["messages"].append({
                "role": "assistant",
                "content": f"✅ Redacted file saved: `{redacted_file_path.name}`"
            })
        else:
            state["messages"].append({
                "role": "assistant",
                "content": f"⚠️ Could not redact file: `{file_path}`"
            })

    return state
