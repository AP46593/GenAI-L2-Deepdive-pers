# This script uses the Ollama library to interact with a language model.
# Ensure you have the Ollama library installed: pip install ollama
# Make sure to run this script in the correct directory where the prompts are located.  


import ollama
from pathlib import Path

# Set the model name you want to use
# Uncomment the model you want to use by removing the '#' at the beginning of the line.
#model_name='phi'
#model_name='llama3:8B'
model_name='deepseek-r1:8B'
#model_name='gemma3n:e4b'
#model_name='gemma3'

# Function to load the prompt from a file

def load_prompt():
    #path = Path(__file__).parent / f"test prompts/Few-Shot.txt"
    path = Path(__file__).parent / f"test prompts/Chain-of-Thoughts.txt"
    #path = Path(__file__).parent / f"test prompts/Tree-of-Thoughts.txt"
    #path = Path(__file__).parent / f"test prompts/Negative-Prompting.txt"
    return path.read_text() if path.exists() else "Hello"



sys_prompt = "You are a helpful assistant."
# Load the prompt from the specified file
# If the file does not exist, it will return a default message "Hello"
prompt1 = load_prompt()

# Print the prompt to verify it has been loaded correctly
print(prompt1)

# Use the Ollama chat function to send the prompt to the model
response = ollama.chat(
    model=model_name,
    messages= [
        {"role":"system","content":sys_prompt},
        {"role":"user","content":prompt1}
               ]
)

# Print the response from the model
print(response["message"]["content"])

# Write the response to a file in the output_test folder
output_path = Path(__file__).parent / "output_test" / "response.txt"
output_path.parent.mkdir(exist_ok=True)  # Create the folder if it doesn't exist

# Write the prompt and response to the output file
with open(output_path, "w", encoding="utf-8") as f:
    f.write(f"{"Prompt Used - "}\n{prompt1}\n\n{"LLM response: "}\n\n{response['message']['content']}")

# Print the path to the output file
print(f"Response written to: {output_path}")
