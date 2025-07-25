
# demonstrate user input to prompt 

from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

# create an instance of ChatOllama with the desired model and temperature
llm = ChatOllama(model="llama3:8B", temperature=0.7)

# define a prompt template for generating a paragraph about a topic
p_templ="Write a short, engaging paragraph about {topic}."
# get user input for the topic
user_txt=input("Enter a topic: ")

# create a PromptTemplate instance from the template string
prompt_tpl = PromptTemplate.from_template(p_templ)
# create a RunnableSequence that combines the prompt template and the LLM
response=llm.invoke(prompt_tpl.format(topic=user_txt))

print(response.content)

