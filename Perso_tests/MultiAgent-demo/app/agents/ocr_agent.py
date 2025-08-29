from pathlib import Path
from app.graph.state import AppState
from app.tools.ocr_utils import extract_text_from_image, extract_urls

def ocr_agent_node(state: AppState) -> AppState:
    files = state.get("uploaded_files", [])
    state["messages"].append({
        "role": "assistant",
        "content": "🔍 Scanning image for text and URLs..."
    })

    results = []

    for file_path in files:
        ext = Path(file_path).suffix.lower()
        if ext not in [".png", ".jpg", ".jpeg"]:
            continue

        text = extract_text_from_image(Path(file_path))
        if not text:
            continue

        urls = extract_urls(text)
        output = f"**Extracted Text:**\n{text.strip()}\n\n"
        if urls:
            output += f"**Detected URLs:**\n" + "\n".join(urls)
        else:
            output += "_No URLs detected._"

        results.append(output)

    if results:
        combined_output = "\n\n---\n\n".join(results)
        state["ocr_output"] = combined_output
        state["messages"].append({
            "role": "assistant",
            "content": combined_output
        })
    else:
        state["messages"].append({
            "role": "assistant",
            "content": "⚠️ No readable images or no text detected."
        })

    return state
