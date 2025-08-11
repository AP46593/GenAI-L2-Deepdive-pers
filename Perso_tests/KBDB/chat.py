"""
Run a local RAG chatbot on your KBDB vector index with a Gradio front‑end.
Start:  python chat.py
Visit:  http://127.0.0.1:7860
"""

from pathlib import Path
import gradio as gr
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import FAISS
import itertools
import time

# ── 1. Load embedder + FAISS index ──────────────────────────────────────────────
INDEX_DIR = Path("faiss_index")
embeddings = OllamaEmbeddings(model="nomic-embed-text")

vectordb = FAISS.load_local(
    str(INDEX_DIR),
    embeddings=embeddings,
    allow_dangerous_deserialization=True,
)

# MMR (Max‑Marginal‑Relevance) retriever: pulls a diverse set of 6 chunks
retriever = vectordb.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 8, "fetch_k": 30},
)

# ── 2. Streaming LLM (Ollama) ───────────────────────────────────────────────────
llm = ChatOllama(model="llama3:8B")  # .stream(...) will be used inside the callback

# ── 3. Helper (generator) for Gradio ────────────────────────────────────────────
def ask_llm(user_query: str, chat_history: list[dict]):
    # 1. Echo the user msg
    chat_history.append({"role": "user", "content": user_query})
    yield chat_history, ""                               # show it now

    spin = itertools.cycle(["⏳ Generating   ", "⏳ Generating.  ", "⏳ Generating.. ", "⏳ Generating..."])

    # 2. Insert *Generating…* bubble so the user sees activity immediately
    assistant_msg = {"role": "assistant", "content": next(spin)}
    chat_history.append(assistant_msg)
    for _ in range(6):           # ~1.5 seconds at 0.25 s per frame
        time.sleep(0.25)
        assistant_msg["content"] = next(spin)
        yield chat_history, ""                             # shows the placeholder

    # 3. Retrieve docs (this can take a second)
    docs = retriever.get_relevant_documents(user_query)
    context = "\n\n---\n\n".join(d.page_content for d in docs)

    # 4. Build RAG prompt
    prompt = (
        "You are a helpful assistant for questions about our HA platform.\n"
        "Answer ONLY from the context below. If unsure, say you don't know.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {user_query}\n\nAnswer:"
    )

    # 5. Stream tokens, replacing the placeholder content
    token_buffer = []
    first_token = True
    for chunk in llm.stream(prompt):
        token_buffer.append(chunk.content)

        # Replace *Generating…* with real text on first token
        if first_token:
            assistant_msg["content"] = "".join(token_buffer)
            first_token = False
        else:
            assistant_msg["content"] = "".join(token_buffer)

        yield chat_history, ""                           # progressive update

    # 6. Append sources footer
    src_set = {f"{d.metadata.get('sheet', '?')} ({d.metadata.get('source', '')})" for d in docs}
    if src_set:
        assistant_msg["content"] += "\n\n**Sources:** " + ", ".join(sorted(src_set))
        yield chat_history, ""                           # final repaint

# ── 4. Gradio UI ────────────────────────────────────────────────────────────────
with gr.Blocks(title="KB RAG Chatbot") as demo:
    gr.Markdown("### 📚 KBDB Chatbot – Ask about your New HA Platform")

    chatbot = gr.Chatbot(
        height=500,
        show_copy_button=True,
        type="messages",   # OpenAI‑style dicts
    )
    msg = gr.Textbox(
        placeholder="Type your question and press Enter …",
        container=False,
        show_label=False,
    )
    clear = gr.ClearButton([msg, chatbot])

    msg.submit(
        fn=ask_llm,            # generator callback
        inputs=[msg, chatbot],
        outputs=[chatbot, msg],
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
