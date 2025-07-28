
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOllama

# A helper function to build conversation history text from memory
def format_history(history):
    # history is a list of (question, answer) tuples
    if not history:
        return "No previous Q&A."
    formatted = ""
    for i, (q, a) in enumerate(history[-5:], start=1):  # last 5 pairs
        formatted += f"Q{i}: {q}\nA{i}: {a}\n"
    return formatted.strip()

# ChatPromptTemplate extended to include conversation history
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

    conversation_history = []  # to store Q&A pairs for current topic

    while True:
        question = input("\nAsk your question (or type 'change topic' or 'exit'): ").strip()
        if question.lower() in ['exit', 'quit']:
            print("Goodbye!")
            run = False
            break
        elif question.lower() == 'change topic':
            break

        # Generate the formatted history text
        history_text = format_history(conversation_history)

        # Invoke chain, passing topic, question & history
        response = chain.invoke({
            "topic": topic,
            "history": history_text,
            "question": question
        })

        answer = response.content
        print("\nAnswer:", answer)

        # Save the Q&A pair to memory
        conversation_history.append((question, answer))
