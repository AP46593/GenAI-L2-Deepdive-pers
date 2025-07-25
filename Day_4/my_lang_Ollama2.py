# demonstrate a reusable prompt template with Ollama's LLM


from langchain_ollama.llms import OllamaLLM
from langchain.prompts import PromptTemplate

def main() -> None:
    llm = OllamaLLM(
        model="llama3:8B",
        base_url="http://127.0.0.1:11434",
        temperature=0.7,
        num_ctx=2048,
        num_predict=128,
    )

    # A reusable template with three placeholders
    template = PromptTemplate(
        input_variables=["tone", "topic", "max_words"],
        template="Write a {tone} quote about {topic}. Keep it under {max_words} words.",
    )

    # Build a RunnableSequence: template ➜ llm
    quote_chain = template | llm          # ← this replaces LLMChain

    # First call
    vars_1 = {"tone": "inspiring", "topic": "AI", "max_words": 12}
    print("Prompt 1 variables:", vars_1)
    print("Llama3  →", quote_chain.invoke(vars_1).strip(), "\n")

    # Second call
    vars_2 = {"tone": "humorous", "topic": "machine learning", "max_words": 15}
    print("Prompt 2 variables:", vars_2)
    print("Llama3  →", quote_chain.invoke(vars_2).strip())

if __name__ == "__main__":
    main()
