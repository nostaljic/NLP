from benchmark import ServiceFactory
import datasets
from transformers import AutoTokenizer, AutoModelForCausalLM

# User should customize to utilize this benchmark
model_directory = "./ai"
model_filename = "qwen.gguf"
dataset_index = 3 # Related to "available_datasets" variables above

# Initialize model and tokenizer
print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_directory, gguf_file=model_filename)
model = AutoModelForCausalLM.from_pretrained(model_directory, gguf_file=model_filename)

available_datasets = [
    ('jaypyon/GSM8K','GSM8K', 'test'),
    ('jaypyon/GSM8K-SOCRATIC','GSM8K', 'test'),
    #----------------------------#
    ('jaypyon/Winogrande_debiased','Winogrande', 'validation'),
    ('jaypyon/Winogrande','Winogrande', 'validation'),
    #----------------------------#
    ('jaypyon/TruthfulMCQA','TruthfulMCQA', 'validation'),
    #----------------------------#
    ('jaypyon/HellaSwag','HellaSwag', 'validation'),
    #----------------------------#
    ('jaypyon/MMLU-Pro','MMLU', 'validation'),
    #----------------------------#
    ('jaypyon/ARC-Challenge','ARC', 'validation'),
    ('jaypyon/ARC-Easy','ARC', 'validation')
]

selected_dataset_name, selected_dataset_type, selected_dataset_split = available_datasets[dataset_index]

# Load dataset
print(f"Loading dataset: {selected_dataset_name}")
dataset = datasets.load_dataset(selected_dataset_name)[selected_dataset_split]

# Get the corresponding service
print(f"Initializing {selected_dataset_type} service...")
service = ServiceFactory.get_service(
    selected_dataset_type, 
    model, 
    tokenizer, 
    dataset
)

# Preprocess the dataset using the service
service.preprocess_dataset()

# Validate the dataset using the service
print("Validating...")
service.validate()
