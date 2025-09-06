from enum import Enum

class ModelEnum(Enum):
    '''
    Baseline: Gemma-2B.
    Core: Gemma-7B, LLaMA-3-8B, Phi-4-14B.
    Stretch: LLaMA-3-13B/14B.
    '''
    GEMMA_2B = "google/gemma-2b"  
    GEMMA_7B = "google/gemma-7b" 
    LLAMA3_8B = "meta-llama/Meta-Llama-3-8B" 
    PHI_4 = "microsoft/phi-4" 

    @classmethod
    def get_model_name(cls, enum_member):
        if isinstance(enum_member, cls):
            return enum_member.value
        raise TypeError(f"Expected a ModelEnum member, but got {type(enum_member)}")