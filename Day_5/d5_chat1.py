import uuid
import gradio as gr

from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables.config import RunnableConfig

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Answer concisely and accurately."),
        MessagesPlaceholder("history", n_messages=8, optional=True),
        ("human", "{question}"),
    ]
)
llm = ChatOllama(model="llama3:8B", temperature=0.7)
base_chain = prompt | llm

_store: dict[str, InMemoryChatMessageHistory] = {}

def get_history(session_id: str):
    return _store.setdefault(session_id, InMemoryChatMessageHistory())

chain = RunnableWithMessageHistory(
    base_chain,
    get_history,
    input_messages_key="question",
    history_messages_key="history",
)

def respond(message, chat_history, session_id):
    # Gradio wants a generator for streaming
    cfg = RunnableConfig(configurable={"session_id": session_id})
    partial = ""
    for chunk in chain.stream({"question": message}, config=cfg):
        partial += chunk.content
        yield chat_history + [(message, partial)]
    # final state yielded implicitly

def new_session():
    # unique session id per browser tab
    return str(uuid.uuid4())

with gr.Blocks() as demo:
    gr.Markdown("## Chat with Llama3:8B (Streaming)")
    session_id = gr.State(new_session())
    chat = gr.Chatbot(height=500)
    msg = gr.Textbox(placeholder="Type a message and press Enter")
    clear = gr.Button("Clear chat (new session)")

    def handle_clear():
        sid = new_session()
        return [], sid

    msg.submit(
        respond,
        inputs=[msg, chat, session_id],
        outputs=[chat],
    )
    msg.submit(lambda: "", None, msg)

    clear.click(handle_clear, outputs=[chat, session_id])

demo.queue().launch()
