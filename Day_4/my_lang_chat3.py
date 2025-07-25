# demonstrate simple chat bot using LangChain and Ollama with session history
# This version uses a session history to maintain context across multiple turns
# It allows the assistant to remember previous questions and answers in the same session

from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory 
from typing import cast
from langchain_core.runnables.config import RunnableConfig

# ───────────────────────── 1. prompt and model ─────────────────────────
prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
         "You are a helpful assistant. Answer concisely and accurately."),
        MessagesPlaceholder("history"),          # whole conversation
        ("human", "{question}")                  # latest user turn
    ]
)
llm = ChatOllama(model="llama3:8B", temperature=0.7)
base_chain = prompt | llm                       # Runnable pipeline

# ──────────────────────── 2. tiny in‑RAM store ────────────────────────
_store: dict[str, InMemoryChatMessageHistory] = {}

def get_history(session_id: str):
    # create once per session
    return _store.setdefault(session_id, InMemoryChatMessageHistory())

chain = RunnableWithMessageHistory(
    base_chain,
    get_history,
    input_messages_key="question",
    history_messages_key="history",
)

# ───────────────────────── 3. console REPL ────────────────────────────
print("Ask me anything (type 'Exit Chat' to quit)\n")
session_id = "local‑session"   # could be user ID, chat‑room ID, etc.

while True:
    try:
        question = input(">>> ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nExiting chat.")
        break

    if question.lower() == "exit chat":
        print("Goodbye!")
        break

    # Always include the configurable session_id
    cfg = cast(RunnableConfig, {"configurable": {"session_id": session_id}})
    

    # Stream the answer
    for chunk in chain.stream({"question": question}, config=cfg):
        print(chunk.content, end="", flush=True)
    print()          # newline after the streamed answer
    print("Next question? type 'Exit Chat' to quit.\n")
# ────────────────────────────── END ────────────────────────────────
