from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig   
import torch
import logging
import os
from models_enum import ModelEnum

# Set up logging for this module
logger = logging.getLogger(__name__)

_MODEL_CACHE = {}

def load_model_and_tokenizer(model_enum: ModelEnum, auth_token: str, local_dir="./models", use_4bit=False):

    if auth_token is None:
        raise ValueError("AUTH_TOKEN not found in environment variables. Make sure .env is loaded.")
    
    model_name = ModelEnum.get_model_name(model_enum)

    if model_name in _MODEL_CACHE:
        logger.info(f"Model '{model_name}' found in cache. Returning cached version.")
        return _MODEL_CACHE[model_name]
    
    # Make sure local directory exists
    os.makedirs(local_dir, exist_ok=True)

    try:
        logger.info(f"Loading tokenizer for '{model_name}' from {local_dir}...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=auth_token,
            cache_dir=local_dir  # ensures persistence
        )
    except Exception as e:
        logger.error(f"Error loading tokenizer: {e}")
        raise

    try:
        logger.info(f"Loading model '{model_name}' from {local_dir}...")
        quant_config = None
        if use_4bit:
            quant_config = BitsAndBytesConfig(load_in_4bit=True)

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=local_dir,
            device_map="auto",              
            torch_dtype=torch.float16,   
            dtype=torch.float16,
            quantization_config=quant_config, 
            token=auth_token,
            resume_download=True,
            # temperature=0.5
        )
        model.eval()

        _MODEL_CACHE[model_name] = (model, tokenizer)
        return model, tokenizer

    except Exception as e:
        logger.error(f"Error loading model '{model_name}': {e}")
        if model_name in _MODEL_CACHE:
            del _MODEL_CACHE[model_name]
        raise

