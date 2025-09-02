import os
import logging
from dotenv import load_dotenv
from model_loader import load_model_and_tokenizer

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load env
load_dotenv()
auth_token = os.getenv("AUTH_TOKEN")

# Model config
MODEL_NAME = "google/gemma-2b"
LOCAL_DIR = "./models/gemma-2b"

# ✅ Load model + tokenizer safely
try:
    model, tokenizer = load_model_and_tokenizer(
        MODEL_NAME,
        auth_token,
        local_dir=LOCAL_DIR,
        use_4bit=True   # set False if you want full precision
    )
    print("✅ Model and tokenizer loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    exit(1)

# --- Test prompt ---
input_text = "Hello, how are you today?"
inputs = tokenizer(input_text, return_tensors="pt").to(next(model.parameters()).device)
outputs = model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))


# # --- PROMPT GENERATION AND INFERENCE ---

# # 1. Set the desired entrance fee
# entrance_fee = 3

# # 2. Generate the prompt using the function
# input_text = get_st_petersburg_prompt(entrance_fee)

# print("--- [Sending Prompt to Model] ---")
# print(input_text)

# # 3. Tokenize the input and move it to the correct device
# inputs = tokenizer(input_text, return_tensors="pt").to(next(model.parameters()).device)

# # 4. Generate the output
# # Note: Increased max_length to allow for a more detailed, reasoned response.
# outputs = model.generate(**inputs, max_length=350) 

# # 5. Decode and print the result
# result = tokenizer.decode(outputs[0], skip_special_tokens=True)

# print("\n--- [Model's Response] ---")
# print(result)



