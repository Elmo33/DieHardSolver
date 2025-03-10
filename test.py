from llama_cpp import Llama

llm = Llama(model_path="mistral-7b-instruct-v0.1.Q4_K_M.gguf")

output = llm("What is 2 + 2?", max_tokens=10)
print(output)
