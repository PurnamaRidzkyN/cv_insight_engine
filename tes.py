from ctransformers import AutoModelForCausalLM

model_path = r"F:\ai_project\cv_analizer\llm\gemma-3-4b-it-q4_0.gguf"
model = AutoModelForCausalLM.from_pretrained(model_path)

prompt = "Hello world"
print(model.generate(prompt, max_new_tokens=20))
