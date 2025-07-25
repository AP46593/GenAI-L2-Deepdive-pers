

# demonstrate live typing with Ollama's LLM

from langchain_ollama import ChatOllama

def main() -> None:
    llm = ChatOllama(model="llama3:8B", temperature=0.7)

    # --- streaming call (live tokens) ---
    print("Streaming haiku:\n")
    for chunk in llm.stream("Write a haiku about clouds"):
        print(chunk.content, end="", flush=True)   # prints each token as soon as it arrives
    print()  # final newline

if __name__ == "__main__":
    main()