
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOllama

# Initialize the local Ollama model
llm = ChatOllama(model="llama3", temperature=0)

# Define the chat prompt template (system + human)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant that answers questions about {topic}."),
        ("human", "{question}")
    ]
)

# Chain prompt and model
chain = prompt | llm

print("Welcome! You can ask questions on any topic using LLaMA3.")
print("Type 'change topic' to switch topics or 'exit' to quit.\n")

run = True
while run:
    topic = input("Enter the topic you'd like to ask about: ")
    if topic.lower() in ['exit', 'quit']:
        print("Goodbye!")
        break

    while True:
        question = input("\nAsk your question (or type 'change topic' or 'exit'): ").strip()
        if question.lower() in ['exit', 'quit']:
            print("Goodbye!")
            run = False
            break
        elif question.lower() == 'change topic':
            break

        # Invoke Ollama with the formatted input
        response = chain.invoke({
            "topic": topic,
            "question": question
        })

        print("\nAnswer:", response.content)
