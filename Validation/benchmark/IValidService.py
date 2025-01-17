import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import datasets
from abc import ABC, abstractmethod

class IValidService(ABC):
    def __init__(
        self,
        model: transformers.AutoModelForCausalLM, 
        tokenizer: transformers.AutoTokenizer, 
        dataset: datasets.Dataset
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.dataset = dataset
        
    def preprocess_dataset(self) -> None:
        self.dataset = self.dataset.map(
            self.preprocess_data, 
            remove_columns=self.dataset.column_names
        )
    
    @abstractmethod
    def validate(self) -> None:
        pass
        
    @abstractmethod
    def preprocess_data(self, row) -> dict:
        pass
        