
import gradio as gr
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOllama

# Initialize model and prompt
llm = ChatOllama(model="llama3", temperature=0)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that answers questions about {topic}."),
    ("human", "{question}")
])

chain = prompt | llm

# Session-level memory
conversation_topic = ""
conversation_history = []

def format_history(history):
    if not history:
        return "No previous Q&A."
    formatted = ""
    for i, (q, a) in enumerate(history[-5:], start=1):
        formatted += f"Q{i}: {q}\nA{i}: {a}\n"
    return formatted.strip()

def chat_interface(topic, question, history_json):
    global conversation_topic, conversation_history

    if topic != conversation_topic:
        conversation_topic = topic
        conversation_history = []

    history_text = format_history(conversation_history)

    # Compose full prompt chain
    response = chain.invoke({
        "topic": topic,
        "question": question
    })

    answer = response.content
    conversation_history.append((question, answer))
    return answer, conversation_history

# Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("# 💬 Ask LLaMA3 Anything")

    topic_input = gr.Textbox(label="Topic", placeholder="e.g. Space, History, Python")
    question_input = gr.Textbox(label="Your Question", placeholder="Ask your question here...")
    chat_output = gr.Textbox(label="Answer")
    state = gr.State([])

    submit_btn = gr.Button("Get Answer")

    submit_btn.click(
        chat_interface,
        inputs=[topic_input, question_input, state],
        outputs=[chat_output, state]
    )

demo.launch()
