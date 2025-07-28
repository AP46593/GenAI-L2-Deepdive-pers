
import os
import json
from datetime import datetime
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOllama

SESSION_DIR = "sessions"
os.makedirs(SESSION_DIR, exist_ok=True)

# Get timestamp-based session ID
def get_session_file(topic):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    topic_clean = topic.lower().replace(" ", "_")
    return os.path.join(SESSION_DIR, f"session_{topic_clean}_{timestamp}.json")

# Format the conversation history for the prompt
def format_history(history):
    if not history:
        return "No previous Q&A."
    formatted = ""
    for i, pair in enumerate(history[-5:], start=1):
        formatted += f"Q{i}: {pair[0]}\nA{i}: {pair[1]}\n"
    return formatted.strip()

# Prompt template
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

print("Welcome! Each session will be saved with a timestamp.")
print("Type 'change topic' to switch or 'exit' to quit.\n")

run = True
while run:
    topic = input("Enter the topic you'd like to ask about: ")
    if topic.lower() in ['exit', 'quit']:
        print("Goodbye!")
        break

    session_history = []
    session_file = get_session_file(topic)

    while True:
        question = input("\nAsk your question (or type 'change topic' or 'exit'): ").strip()
        if question.lower() in ['exit', 'quit']:
            with open(session_file, 'w') as f:
                json.dump(session_history, f)
            print("Goodbye!")
            run = False
            break
        elif question.lower() == 'change topic':
            with open(session_file, 'w') as f:
                json.dump(session_history, f)
            break

        history_text = format_history(session_history)

        response = chain.invoke({
            "topic": topic,
            "history": history_text,
            "question": question
        })

        answer = response.content
        print("\nAnswer:", answer)

        session_history.append((question, answer))
