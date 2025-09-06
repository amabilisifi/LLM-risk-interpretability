from enum import Enum

class ModelEnum(Enum):
    '''
    Baseline: Gemma-2B.
    Core: Gemma-7B, LLaMA-3-8B, Phi-4-14B.
    Stretch: LLaMA-3-13B/14B.
    '''
     # --- Baseline ---
    GEMMA_2B = "google/gemma-2b"  

    # --- Core Models ---
    GEMMA_7B = "google/gemma-7b" 
    LLAMA3_8B = "meta-llama/Meta-Llama-3-8B" 
    PHI_4 = "microsoft/phi-4" 

    # --- Stretch Models (optional, if GPU allow) ---
    LLAMA3_13B = "meta-llama/Meta-Llama-3-13B"
    # LLAMA3_27B = "meta-llama/Meta-Llama-3-27B"
    # LLAMA3_70B = "meta-llama/Meta-Llama-3-70B"

    @classmethod
    def get_model_name(cls, enum_member):
        if isinstance(enum_member, cls):
            return enum_member.value
        raise TypeError(f"Expected a ModelEnum member, but got {type(enum_member)}")