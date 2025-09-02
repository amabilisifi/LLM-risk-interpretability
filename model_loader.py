from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig   
import torch
import logging
import os

# Set up logging for this module
logger = logging.getLogger(__name__)

def load_model_and_tokenizer(model_name, auth_token, local_dir="./models/gemma-7b", use_4bit=False):

    if auth_token is None:
        raise ValueError("AUTH_TOKEN not found in environment variables. Make sure .env is loaded.")

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

        # Optional quantization to reduce VRAM usage
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
            resume_download=True  # Force resume from existing files
        )

        model.eval()
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

    return model, tokenizer