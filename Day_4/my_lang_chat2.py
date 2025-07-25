
# demonstrate simple chat bot using LangChain and Ollama - no memory/history
# This is a simplified version of the chat bot that does not use any session history or memory

from langchain.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

def main() -> None:
    # Build prompt template and LLM only once
    prompt = ChatPromptTemplate.from_template(
        "You are a helpful assistant. You will be given a question, "
        "and you should provide a concise and accurate answer based on the information provided.\n"
        "Question: {question}\n"
        "Answer:"
    )
    llm = ChatOllama(model="llama3:8B", temperature=0.7)
    chain = prompt | llm        # Runnable pipeline

    print("Ask me anything (type “Exit Chat” to quit).\n")

    while True:
        try:
            user_q = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            # Handles Ctrl‑D / Ctrl‑C gracefully
            print("\nExiting chat.")
            break

        if user_q.lower() == "exit chat":
            print("Goodbye!")
            break

        # Stream the answer
        for chunk in chain.stream({"question": user_q}):
            print(chunk.content, end="", flush=True)
        print()  # newline after the streamed answer
        print("Next question? type “Exit Chat” to quit).\n")
        print()

if __name__ == "__main__":
    main()