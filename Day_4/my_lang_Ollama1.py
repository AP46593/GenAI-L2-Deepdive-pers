from langchain_ollama.llms import OllamaLLM      # ← NEW import

def main() -> None:
    llm = OllamaLLM(
        model="llama3:8B",                 # name shown by `ollama list`
        base_url="http://127.0.0.1:11434",
        temperature=0.7,                # normal Ollama parameters…
        num_ctx=2048,                   # context window      (was n_ctx)
        num_predict=128,                # max output tokens   (was max_length)
    )

    reply = llm.invoke("Write a one‑line motivational quote about AI.")
    print("AI →", reply.strip())

if __name__ == "__main__":
    main()