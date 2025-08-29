from pathlib import Path
from app.graph.state import AppState
from app.tools.faiss_store import save_to_faiss
from langchain.document_loaders import PyMuPDFLoader, UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

VECTORSTORE_DIR = Path("data/vectorstore")
VECTORSTORE_DIR.mkdir(exist_ok=True, parents=True)

def rag_agent_node(state: AppState) -> AppState:
    files = state.get("uploaded_files", [])
    all_docs = []

    for file_path in files:
        ext = Path(file_path).suffix.lower()
        try:
            if ext == ".pdf":
                loader = PyMuPDFLoader(file_path)
            elif ext == ".docx":
                loader = UnstructuredWordDocumentLoader(file_path)
            else:
                continue

            docs = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            split_docs = splitter.split_documents(docs)
            all_docs.extend(split_docs)

        except Exception as e:
            state["messages"].append({
                "role": "assistant",
                "content": f"❌ Failed to process `{file_path}`: {str(e)}"
            })

    if all_docs:
        vectorstore_path = VECTORSTORE_DIR / "kb_index"
        save_to_faiss(all_docs, str(vectorstore_path))
        state["vector_store_path"] = str(vectorstore_path)
        state["messages"].append({
            "role": "assistant",
            "content": f"✅ Successfully ingested {len(all_docs)} chunks into the knowledge base!"
        })
    else:
        state["messages"].append({
            "role": "assistant",
            "content": "⚠️ No supported documents were ingested."
        })

    return state
