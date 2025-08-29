from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
from dotenv import load_dotenv
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

model_name = "google/gemma-2b"
auth_token = os.getenv('AUTH_TOKEN')
# print("Auth Token:", auth_token)
if auth_token is None:
    raise ValueError("AUTH_TOKEN not found in environment variables.")

try:
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=auth_token)
    print("Tokenizer loaded successfully.")
except Exception as e:
    logger.error(f"Error loading tokenizer: {e}")
    exit(1)

try:
    logger.info("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        token=auth_token
    )
    model.eval()
    print("Model loaded successfully.")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    exit(1)

# input_text = "Hello, how can I help you today?"
# inputs = tokenizer(input_text, return_tensors="pt").to(next(model.parameters()).device)
# outputs = model.generate(**inputs, max_length=50)
# print(tokenizer.decode(outputs[0], skip_special_tokens=True))