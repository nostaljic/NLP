import json
from .llm_service import LLMService
from transformers import AutoTokenizer, AutoModelForCausalLM

class GGUFLLMService(LLMService):
    def __init__(self, model_path: str, model_filename: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, gguf_file=model_filename)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, gguf_file=model_filename)
    
    def generate_response(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(inputs["input_ids"], temperature=1.0, max_new_tokens=100, use_cache=True)#, num_return_sequences=1)
        return self.tokenizer.decode(outputs[0]).split("<|im_start|>assistant")[1].split("<|im_end|>")[0]
    
    def load_prompt(self, file_path: str, key:str) -> str:
        with open(file_path, "r") as file:
            data = json.load(file)
        return data.get(key, "")
