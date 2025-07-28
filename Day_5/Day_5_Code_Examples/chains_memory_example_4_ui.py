
import gradio as gr
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage
from langchain_community.chat_models import ChatOllama
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from datetime import datetime

# Prompt and memory
prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="You are a helpful assistant."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

memory = ConversationBufferMemory(memory_key="history", return_messages=True)
llm = ChatOllama(model="llama3", temperature=0)

conversation = ConversationChain(
    llm=llm,
    prompt=prompt,
    memory=memory,
    verbose=True
)

# Chat interface function
def chat(input_text, history=[]):
    if input_text.lower() in ["exit", "quit"]:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"chat_memory_{timestamp}.txt"
        with open(filename, "w") as f:
            for msg in memory.chat_memory.messages:
                line = f"[{msg.type.upper()}] {msg.content}"
                f.write(line + "\n")
        return "Session ended. Memory saved to file.", history

    response = conversation.invoke({"input": input_text})
    history.append((input_text, response["response"]))
    return response["response"], history

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("# 🧠 LLaMA3 Chat Assistant with Memory")
    chatbot = gr.Chatbot()
    msg = gr.Textbox(label="Enter your question")
    clear = gr.Button("Clear Chat")

    state = gr.State([])

    def user_submit(user_message, chat_history):
        response, chat_history = chat(user_message, chat_history)
        return "", chat_history

    msg.submit(user_submit, [msg, state], [msg, chatbot])
    clear.click(lambda: ([], []), None, [state, chatbot])

demo.launch()
