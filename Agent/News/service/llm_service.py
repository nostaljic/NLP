from abc import ABC, abstractmethod

class LLMService(ABC):
    @abstractmethod
    def generate_response(self, prompt: str) -> str:
        pass

    @abstractmethod
    def load_prompt(self, file_path: str) -> str:
        pass
