from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
from dotenv import load_dotenv

load_dotenv()

model_name = "google/gemma-2b"
auth_token = os.getenv('AUTH_TOKEN')
print(auth_token)
tokenizer = AutoTokenizer.from_pretrained(model_name, token=auth_token)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16,
    load_in_4bit=True,
    token=auth_token
)
model.eval()
