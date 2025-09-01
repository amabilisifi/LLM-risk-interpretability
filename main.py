from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
from dotenv import load_dotenv
import logging

from prompts import get_st_petersburg_prompt

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

# --- PROMPT GENERATION AND INFERENCE ---

# 1. Set the desired entrance fee
entrance_fee = 3

# 2. Generate the prompt using the function
input_text = get_st_petersburg_prompt(entrance_fee)

print("--- [Sending Prompt to Model] ---")
print(input_text)

# 3. Tokenize the input and move it to the correct device
inputs = tokenizer(input_text, return_tensors="pt").to(next(model.parameters()).device)

# 4. Generate the output
# Note: Increased max_length to allow for a more detailed, reasoned response.
outputs = model.generate(**inputs, max_length=350) 

# 5. Decode and print the result
result = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("\n--- [Model's Response] ---")
print(result)


# from dotenv import load_dotenv
# import logging
# from model_loader import load_model_and_tokenizer # <-- IMPORT your new function

# # This should be the very first thing to run
# load_dotenv() 

# # Set up logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# def get_st_petersburg_prompt(entrance_fee):
#     """
#     Generates a prompt describing the St. Petersburg paradox game.
#     """
#     return f"""You are offered to play a game... (rest of your prompt string)""" # Truncated for brevity

# # --- LOAD MODEL (Now just one function call!) ---
# logging.info("Attempting to load model and tokenizer...")
# model, tokenizer = load_model_and_tokenizer()
# logging.info("Model and tokenizer loaded successfully.")

# # --- PROMPT GENERATION AND INFERENCE ---
# entrance_fee = 3
# input_text = get_st_petersburg_prompt(entrance_fee)

# logging.info("Generating response...")
# inputs = tokenizer(input_text, return_tensors="pt").to(next(model.parameters()).device)
# outputs = model.generate(**inputs, max_length=350) 
# result = tokenizer.decode(outputs[0], skip_special_tokens=True)

# print("\n--- [Model's Response] ---")
# print(result)