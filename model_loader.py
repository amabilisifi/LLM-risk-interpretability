from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
import logging

# Set up logging for this module
logger = logging.getLogger(__name__)

def load_model_and_tokenizer():
    """
    Loads the Gemma-2b model and tokenizer from Hugging Face.
    Handles authentication and device mapping.
    """
    model_name = "google/gemma-2b"
    auth_token = os.getenv('AUTH_TOKEN')
    
    if auth_token is None:
        raise ValueError("AUTH_TOKEN not found in environment variables. Make sure .env file is loaded.")

    try:
        logger.info(f"Loading tokenizer for '{model_name}'...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=auth_token)
    except Exception as e:
        logger.error(f"Error loading tokenizer: {e}")
        raise # Re-raise the exception to stop the program

    try:
        logger.info(f"Loading model '{model_name}'...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            token=auth_token
        )
        model.eval()
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise # Re-raise the exception

    return model, tokenizer