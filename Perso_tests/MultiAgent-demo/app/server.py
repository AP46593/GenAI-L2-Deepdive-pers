import gradio as gr
from pathlib import Path
from app.graph.build_graph import build_graph
from app.graph.state import AppState

chat_graph = build_graph()

state: AppState = {
    "messages": [],
    "task_intent": "",
    "uploaded_files": []
}

UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

def handle_user_input(user_msg, files):
    if files:
        for f in files:
            target_path = UPLOAD_DIR / Path(f.name).name
            with open(target_path, "wb") as out:
                out.write(f.read())
            state["uploaded_files"].append(str(target_path))

    state["messages"].append({"role": "user", "content": user_msg})

    new_state = chat_graph.invoke(state)
    messages = [(m["role"], m["content"]) for m in new_state["messages"]]

    # check for redacted file
    redacted_path = new_state.get("redacted_file")
    download_path = redacted_path if redacted_path and Path(redacted_path).exists() else None

    return messages, download_path

chatbot_ui = gr.Chatbot(label="Agentic Multi-Agent Demo")

with gr.Blocks() as demo:
    gr.Markdown("### 🤖 LangGraph + MCP Chat Demo")

    chat = chatbot_ui
    msg = gr.Textbox(placeholder="Ask me anything or upload a file...", label="Your message")
    upload = gr.File(file_types=[".pdf", ".docx", ".png", ".jpg", ".jpeg"], file_count="multiple")

    # ✅ define only once
    download = gr.File(label="Download redacted document", visible=False, interactive=True)

    def update_ui(user_msg, files):
        chat_msgs, download_path = handle_user_input(user_msg, files)
        if download_path:
            return chat_msgs, None, download_path, download.update(visible=True)   # type: ignore
        else:
            return chat_msgs, None, None, download.update(visible=False)   # type: ignore

    # ✅ define outputs using component instance `download`
    msg.submit(update_ui, [msg, upload], [chat, upload, download, download])
    upload.change(update_ui, [msg, upload], [chat, upload, download, download])

demo.launch()
