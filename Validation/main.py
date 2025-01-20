from benchmark import ArcValidService
import datasets
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "./ai"
filename = "qwen.gguf"

tokenizer = AutoTokenizer.from_pretrained(model_id, gguf_file=filename)
model = AutoModelForCausalLM.from_pretrained(model_id, gguf_file=filename)
#dataset = datasets.load_dataset('allenai/ai2_arc', 'ARC-Challenge')['train']
#dataset = datasets.load_dataset('allenai/ai2_arc', 'ARC-Challenge')['test']
dataset = datasets.load_dataset('allenai/ai2_arc', 'ARC-Challenge')['validation']
arc_val = ArcValidService(model,tokenizer,dataset)
arc_val.preprocess_dataset()
arc_val.validate()