import os
import logging
import json
import torch
from dotenv import load_dotenv
from datetime import datetime

from model_loader import load_model_and_tokenizer
from models_enum import ModelEnum
from prompts import get_st_petersburg_prompt
from output_utils import generate_json_output
from logit_lens import apply_logit_lens, analyze_logit_lense  # Note: fix import if filename is logit_lense.py

# --- Logging setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Load env ---
load_dotenv()
auth_token = os.getenv("AUTH_TOKEN")

# --- Model setup ---
MODEL_NAME = ModelEnum.GEMMA_2B
LOCAL_DIR = "./models/" + ModelEnum.get_model_name(MODEL_NAME)

try:
    # If model_loader doesn't already have it, add attn_implementation="eager" here or in model_loader.py
    # Example: model = AutoModelForCausalLM.from_pretrained(..., attn_implementation="eager")
    model, tokenizer = load_model_and_tokenizer(
        MODEL_NAME,
        auth_token,
        local_dir=LOCAL_DIR,
        use_4bit=True
    )
    print("Model and tokenizer loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    exit(1)

# --- Output directory ---
output_dir = "./outputs"
os.makedirs(output_dir, exist_ok=True)

# --- Prompt ---
entrance_fee = 100
input_text = get_st_petersburg_prompt(entrance_fee)
print("--- [Sending Prompt to Model] ---")
print(input_text)

inputs = tokenizer(input_text, return_tensors="pt").to(next(model.parameters()).device)

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# --- Run generation ---
try:
    outputs = model.generate(
        **inputs,
        do_sample=False,
        output_scores=True,
        return_dict_in_generate=True,
        top_p=1.0,
        max_new_tokens=180,
        repetition_penalty=1.1,
        output_attentions=True,
        output_hidden_states=True,
    )
    # Decode ONLY new tokens (fix for seeing only prompt)
    input_len = inputs['input_ids'].shape[1]
    result = tokenizer.decode(outputs.sequences[0][input_len:], skip_special_tokens=True)
except Exception as e:
    logger.error(f"Failed to generate output: {e}")
    result = "Fallback: generation failed."

# --- Save raw tensors (weights, attn, hidden states) ---
output_path = f"{output_dir}/weights_attn_{MODEL_NAME.name}_{timestamp}.pt"
try:
    torch.save({
        "sequences": outputs.sequences,
        "scores": outputs.scores,
        "attentions": outputs.attentions,
        "hidden_states": outputs.hidden_states
    }, output_path)
except Exception as e:
    logger.error(f"Failed to save torch tensors: {e}")

# --- Apply logit lens ---
try:
    print("\n--- [Logit Lens Analysis] ---")
    res = apply_logit_lens(model, tokenizer, output_path)  # Fixed: pass file path, not sequences
    print(res)
    print("kokokok")
    ana = analyze_logit_lense(res, timestamp)  # Added timestamp for dynamic JSON/plot naming
    print(ana)
    print("jojojoji")
except Exception as e:
    logger.error(f"Failed to apply logit lens: {e}")

# --- Save JSON output ---
json_output = generate_json_output(result, entrance_fee)
output_filename = f"{output_dir}/st_petersburg_{MODEL_NAME.name}_{timestamp}.json"
try:
    with open(output_filename, 'w') as f:
        json.dump(json_output, f, indent=4)
    print(f"\nJSON output saved to {output_filename}")
except Exception as e:
    logger.error(f"Failed to save JSON output: {e}")

# --- Save metadata ---
try:
    num_layers = len(outputs.hidden_states) - 1 if outputs.hidden_states else 0  # Adjust for initial embed layer
    heads_per_layer = None
    if outputs.attentions and outputs.attentions[0] is not None and len(outputs.attentions) > 0:
        heads_per_layer = outputs.attentions[0][0].shape[1]  # Fixed: check None, index safely

    meta = {
        "model": MODEL_NAME.name,
        "timestamp": datetime.now().isoformat(),
        "num_tokens": outputs.sequences.shape[-1] if outputs.sequences is not None else 0,
        "num_layers": num_layers,
        "heads_per_layer": heads_per_layer,
    }

    with open(f"{output_dir}/weights_attn_meta_{MODEL_NAME.name}_{timestamp}.json", "w") as f:
        json.dump(meta, f, indent=4)
except Exception as e:
    logger.error(f"Failed to save metadata: {e}")

# --- Print final output ---
print("\n--- [Model's Response] ---")
print(result)