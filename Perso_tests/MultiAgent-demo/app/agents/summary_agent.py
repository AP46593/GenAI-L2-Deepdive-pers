from app.graph.state import AppState
from app.tools.faiss_store import load_faiss
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_ollama import ChatOllama

LLM = ChatOllama(model="llama3")  # or "mistral" etc.

SUMMARY_PROMPT = PromptTemplate.from_template("""
You are a summarizer agent. The user has uploaded one or more documents. 
You are given relevant chunks of the documents. Summarize the main ideas, key points, and insights in under 500 words.
Ignore boilerplate or legal disclaimers.

Documents:
{context}

Return only the summary.
""")

def doc_summary_agent_node(state: AppState) -> AppState:
    messages = state["messages"]
    vector_path = state.get("vector_store_path")
    if not vector_path:
        state["messages"].append({
            "role": "assistant",
            "content": "⚠️ No document has been ingested yet. Please upload and ingest a file first."
        })
        return state

    try:
        vectorstore = load_faiss(vector_path)
        retriever = vectorstore.as_retriever(search_type="similarity", k=6)

        # Use last user message as the query
        query = messages[-1]["content"]

        qa = RetrievalQA.from_chain_type(
            llm=LLM,
            retriever=retriever,
            return_source_documents=False,
            chain_type="stuff",
            chain_type_kwargs={"prompt": SUMMARY_PROMPT}
        )

        result = qa.run(query)
        state["doc_summary"] = result
        state["messages"].append({
            "role": "assistant",
            "content": f"📄 Here's a summary of your uploaded documents:\n\n{result}"
        })

    except Exception as e:
        state["messages"].append({
            "role": "assistant",
            "content": f"❌ Failed to summarize document: {str(e)}"
        })

    return state
