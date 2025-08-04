import os, json, tempfile, textwrap
import gradio as gr
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, WebBaseLoader
from langchain.schema import Document


# ────────────────────────────────────────────────────────────
# API key (set as environment variable or in .env file)
# ────────────────────────────────────────────────────────────
# os.environ["OPENAI_API_KEY"] = "your-api-key-here"  # Uncomment and set your API key

# fetch api key from environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────
def _format_docs(docs):
    """Format retrieved docs with simple citations."""
    lines = []
    for d in docs:
        src = d.metadata.get("source") or d.metadata.get("url") or "unknown"
        page = d.metadata.get("page", d.metadata.get("chunk_id", "na"))
        lines.append(f"[{src}#{page}] {d.page_content}")
    return "\n\n".join(lines)

def _citations(docs):
    """Return a compact citations string like [source#page], deduped."""
    tags = []
    seen = set()
    for d in docs:
        src = d.metadata.get("source") or d.metadata.get("url") or "unknown"
        page = d.metadata.get("page", d.metadata.get("chunk_id", "na"))
        tag = f"[{src}#{page}]"
        if tag not in seen:
            tags.append(tag)
            seen.add(tag)
    return " ".join(tags) if tags else "(no citations)"

# ────────────────────────────────────────────────────────────
# Build index: load → chunk → embed → FAISS (cosine)
# ────────────────────────────────────────────────────────────
def build_index(
    source_type,
    files,        # list of temp files from gradio for "Files"
    urls_text,    # str for "URL(s)" separated by newline/comma
    raw_text,     # str for "Raw Text"
    chunk_size,
    chunk_overlap,
    k,
    model_name,
    temperature
):
    try:
        # 1) Load docs
        docs = []
        if source_type == "Files (PDF/TXT)":
            if not files:
                return gr.update(value="Please upload at least one file."), None, None, None
            for f in files:
                # Decide by extension
                name = f.name if hasattr(f, "name") else str(f)
                if name.lower().endswith(".pdf"):
                    loader = PyPDFLoader(name)
                    docs.extend(loader.load())
                else:
                    loader = TextLoader(name, encoding="utf-8")
                    docs.extend(loader.load())

        elif source_type == "URL(s)":
            if not urls_text.strip():
                return gr.update(value="Please enter at least one URL."), None, None, None
            # split on newline or comma
            raw_urls = [u.strip() for u in urls_text.replace(",", "\n").split("\n") if u.strip()]
            loader = WebBaseLoader(raw_urls)
            docs = loader.load()

        elif source_type == "Raw Text":
            if not raw_text.strip():
                return gr.update(value="Please paste some text."), None, None, None
            docs = [Document(page_content=raw_text, metadata={"source": "raw.txt"})]

        else:
            return gr.update(value="Unsupported source type."), None, None, None

        if not docs:
            return gr.update(value="No text found after loading."), None, None, None

        # 2) Chunk
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=int(chunk_size), chunk_overlap=int(chunk_overlap)
        )
        splits = splitter.split_documents(docs)

        # 3) Embeddings + FAISS (cosine)
        emb = OpenAIEmbeddings(model="text-embedding-3-small")
        vs = FAISS.from_documents(splits, embedding=emb, distance_strategy=DistanceStrategy.COSINE)
        retriever = vs.as_retriever(search_kwargs={"k": int(k)})

        # 4) LLM (GPT model)
        llm = ChatOpenAI(model=model_name, temperature=float(temperature))

        status = (
            f"✅ Index built.\n"
            f"- Chunks: {len(splits)}\n"
            f"- k: {k}\n"
            f"- LLM: {model_name} (temp={temperature})\n"
            f"- Similarity: COSINE (FAISS)"
        )
        return status, vs, retriever, llm

    except Exception as e:
        return f"❌ Error while building index: {e}", None, None, None

# ────────────────────────────────────────────────────────────
# Answer a question from the retriever context
# ────────────────────────────────────────────────────────────
def ask_question(chat_history, question, retriever, llm):
    if retriever is None or llm is None:
        chat_history = chat_history or []
        chat_history.append(("",
            "Please build the index first (choose a source and click 'Build Index')."
        ))
        return chat_history

    try:
        # Retrieve
        ctx_docs = retriever.get_relevant_documents(question)
        context = _format_docs(ctx_docs)

        # Prompt (simple & explicit grounding)
        messages = [
            {"role": "system",
             "content": "You are a helpful assistant. Use ONLY the provided context to answer. "
                        "If the answer is not present, say you don't know."},
            {"role": "user",
             "content": f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"}
        ]
        resp = llm.invoke(messages).content

        # Append citations at the end
        cites = _citations(ctx_docs)
        final = f"{resp}\n\n**Citations:** {cites}"

        chat_history = chat_history or []
        chat_history.append((question, final))
        return chat_history

    except Exception as e:
        chat_history = chat_history or []
        chat_history.append((question, f"❌ Error: {e}"))
        return chat_history

# ────────────────────────────────────────────────────────────
# Gradio UI
# ────────────────────────────────────────────────────────────
with gr.Blocks(title="Tiny RAG UI (FAISS + OpenAI)") as app:
    gr.Markdown("## 🔎 Tiny RAG — Ask questions over your PDF/URL/TXT/Raw text")

    with gr.Accordion("1) Build Index", open=True):
        source_type = gr.Radio(
            ["Files (PDF/TXT)", "URL(s)", "Raw Text"],
            value="Files (PDF/TXT)",
            label="Source type"
        )
        files = gr.File(label="Upload PDF/TXT files", file_types=[".pdf", ".txt"], file_count="multiple", visible=True)
        urls_text = gr.Textbox(label="Enter URL(s) (comma- or newline‑separated)", visible=False, lines=3)
        raw_text = gr.Textbox(label="Paste Raw Text", visible=False, lines=6, placeholder="Paste your text here...")

        with gr.Row():
            chunk_size = gr.Number(label="Chunk size", value=800)
            chunk_overlap = gr.Number(label="Chunk overlap", value=120)
            k = gr.Slider(1, 10, value=3, step=1, label="Top‑k retrieved")

        with gr.Row():
            model_name = gr.Textbox(label="GPT model", value="gpt-4o-mini")
            temperature = gr.Slider(0.0, 1.0, value=0.2, step=0.05, label="Temperature")

        build_btn = gr.Button("🔨 Build Index")
        status = gr.Markdown("Status will appear here.")

    with gr.Accordion("2) Ask Questions", open=True):
        chat = gr.Chatbot(height=320, label="Chat")
        question = gr.Textbox(label="Your question")
        ask_btn = gr.Button("Ask")

    # App state
    vs_state = gr.State(None)
    retriever_state = gr.State(None)
    llm_state = gr.State(None)

    # Source type toggling
    def toggle_inputs(src):
        return (
            gr.update(visible=(src == "Files (PDF/TXT)")),
            gr.update(visible=(src == "URL(s)")),
            gr.update(visible=(src == "Raw Text")),
        )

    source_type.change(
        toggle_inputs, inputs=[source_type], outputs=[files, urls_text, raw_text]
    )

    # Build index
    build_btn.click(
        build_index,
        inputs=[source_type, files, urls_text, raw_text, chunk_size, chunk_overlap, k, model_name, temperature],
        outputs=[status, vs_state, retriever_state, llm_state]
    )

    # Ask
    ask_btn.click(
        ask_question,
        inputs=[chat, question, retriever_state, llm_state],
        outputs=[chat]
    )

# Launch (share=True gives a public URL from Colab)
app.launch(share=True)