
# demonstrate langchain_core prompts with Ollama's LLM

from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

# create an instance of ChatOllama with the desired model and temperature
llm = ChatOllama(model="llama3:8B", temperature=0.7)

# define a prompt template for generating a paragraph about a topic
prompt_tpl = PromptTemplate.from_template(
    "Write a short, engaging paragraph about {topic}."
)

# create a RunnableSequence that combines the prompt template and the LLM
chain = prompt_tpl | llm

# invoke the chain with a specific topic
print(chain.invoke({"topic": " Meta - Facebook "}).content)