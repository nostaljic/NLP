#Valid
from .IValidService import IValidService
from .ArcValidService import ArcValidService
from .MMLUValidService import MMLUValidService
from .HellaSwagValidService import HellaSwagValidService
from .TruthfulMCQAValidService import TruthfulMCQAValidService
from .WinograndeValidService import WinograndeValidService
from .GSM8KValidService import GSM8KValidService


class ServiceFactory:
    services = {
        "ARC": ArcValidService,
        "MMLU": MMLUValidService,
        "HellaSwag": HellaSwagValidService,
        "TruthfulMCQA": TruthfulMCQAValidService,
        "Winogrande": WinograndeValidService,
        "GSM8K": GSM8KValidService
    }

    @staticmethod
    def get_service(service_name: str, model, tokenizer, dataset) -> IValidService:
        service_class = ServiceFactory.services.get(service_name)
        if service_class is None:
            raise ValueError(f"Service '{service_name}' is not registered.")
        return service_class(model, tokenizer, dataset)

# Expose all services and factory from the package
__all__ = [
    "ArcValidService",
    "MMLUValidService",
    "HellaSwagValidService",
    "TruthfulMCQAValidService",
    "WinograndeValidService",
    "GSM8KValidService",
    "ServiceFactory",
]