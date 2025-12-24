Install VSCODE, Python, Ollama ( pull llama3:3B manifest)


Change the model in the py files - if you want to try other models instead of llama3:8B

For huggingface/openAi models - add a .env file with your api_keys in your project.


Do pip install for the required packages like openai, langchain, etc - depending on the py file you are trying to run.



Ollama guide:

#open a terminal window and run 
ollama serve
#this will start ollama server 
#keep this terminal active and open a separate terminal for everything else

#pull model llama3.2:latest for chat and generic llm operations
ollama pull llama3.2:latest

#pull model nomic-embed-text:latest for creating embeddings for RAG
ollama pull nomic-embed-text:latest

#check list of downloaded ollama models 
ollama list 

#check for running models
ollama ps


