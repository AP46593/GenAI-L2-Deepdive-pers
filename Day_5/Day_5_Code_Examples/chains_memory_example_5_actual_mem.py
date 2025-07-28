
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

# Set up the prompt (no need for manual history insertion now)
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{input}")
])

# Set up memory to track conversation context
memory = ConversationBufferMemory(return_messages=True)

# Initialize the local LLaMA3 model
llm = ChatOllama(model="llama3", temperature=0)

# Build the conversation chain with memory
conversation = ConversationChain(
    llm=llm,
    prompt=prompt,
    memory=memory,
    verbose=True
)

print("Welcome! You can chat with LLaMA3. Type 'exit' to quit.\n")

while True:
    user_input = input("You: ").strip()
    if user_input.lower() in ['exit', 'quit']:
        print("Goodbye!")
        break

    response = conversation.invoke({"input": user_input})
    print("\nAssistant:", response["response"])
