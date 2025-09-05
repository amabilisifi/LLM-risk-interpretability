from enum import Enum

class ModelEnum(Enum):
    GEMMA_2B = "google/gemma-2b" # for trying and initial stuff - baseline
    GEMMA_7B = "google/gemma-7b" # maybe later for comparing gemma
    LLAMA3_8B = "meta-llama/Meta-Llama-3-8B" # main model 1
    MISTRAL_7B = "mistralai/Mistral-7B-v0.1"  # main model 2
    # maybe later optional phi-4

    @classmethod
    def get_model_name(cls, enum_member):
        if isinstance(enum_member, cls):
            return enum_member.value
        raise TypeError(f"Expected a ModelEnum member, but got {type(enum_member)}")