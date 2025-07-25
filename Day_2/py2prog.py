import ollama
 
 
response = ollama.chat(
    model='llama3:8B',
    messages= [
        {"role":"system","content":"you are a python programmer"},
        {"role":"user","content":"Write a python program to print 'Hello, World!'."}
               ]
)
 
print(type(response))
 
print(response['message']['content'])