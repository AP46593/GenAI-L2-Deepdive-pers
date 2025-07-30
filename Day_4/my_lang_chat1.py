from langchain.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama


# demonstrate user input to prompt for a question and answer format
p_template=( "You are a helpful assistant. You will be given a question, and you should provide a concise and accurate answer based on the information provided. \n"
            "Question: {question}\n"
            "Answer: "
            )

# create a ChatPromptTemplate instance from the template string
prompt=ChatPromptTemplate.from_template(p_template)

user_input=input("Please enter your question: ")
full_prompt=prompt.format(question=user_input)

llm = ChatOllama(model="llama3", temperature=0.7)

# create a RunnableSequence that combines the prompt template and the LLM
# Invoke the LLM with the formatted prompt
#esponse=llm.invoke(full_prompt)
# Print the response content
#rint("Response from Llama3 : \n", response.content)

for chunk in llm.stream(full_prompt):              # <- live chunks
    print(chunk.content, end="", flush=True)       # token(s) as they arrive
print() 
