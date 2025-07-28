
import os
import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOllama

HISTORY_DIR = "history"
os.makedirs(HISTORY_DIR, exist_ok=True)

# Load history from a JSON file
def load_history(topic):
    filepath = os.path.join(HISTORY_DIR, f"history_{topic.lower().replace(' ', '_')}.json")
    if os.path.exists(filepath):
        with open(filepath, 'r') as file:
            return json.load(file)
    return []

# Save history to a JSON file
def save_history(topic, history):
    filepath = os.path.join(HISTORY_DIR, f"history_{topic.lower().replace(' ', '_')}.json")
    with open(filepath, 'w') as file:
        json.dump(history, file)

# Format the conversation history for the prompt
def format_history(history):
    if not history:
        return "No previous Q&A."
    formatted = ""
    for i, pair in enumerate(history[-5:], start=1):
        formatted += f"Q{i}: {pair[0]}\nA{i}: {pair[1]}\n"
    return formatted.strip()

# Prompt with dynamic history
prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
         "You are a helpful assistant that answers questions about {topic}.\n"
         "Here is the recent conversation history:\n"
         "{history}\n"
         "Answer the next question carefully."),
        ("human", "{question}")
    ]
)

llm = ChatOllama(model="llama3", temperature=0)
chain = prompt | llm

print("Welcome! You can ask questions on any topic using LLaMA3.")
print("Type 'change topic' to switch topics or 'exit' to quit.\n")

run = True
while run:
    topic = input("Enter the topic you'd like to ask about: ")
    if topic.lower() in ['exit', 'quit']:
        print("Goodbye!")
        break

    conversation_history = load_history(topic)

    while True:
        question = input("\nAsk your question (or type 'change topic' or 'exit'): ").strip()
        if question.lower() in ['exit', 'quit']:
            print("Goodbye!")
            run = False
            break
        elif question.lower() == 'change topic':
            save_history(topic, conversation_history)
            break

        history_text = format_history(conversation_history)

        response = chain.invoke({
            "topic": topic,
            "history": history_text,
            "question": question
        })

        answer = response.content
        print("\nAnswer:", answer)

        conversation_history.append((question, answer))
        save_history(topic, conversation_history)
