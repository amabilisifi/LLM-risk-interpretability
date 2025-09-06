import os
import logging
import json
import torch
from dotenv import load_dotenv
from datetime import datetime

from model_loader import load_model_and_tokenizer
from models_enum import ModelEnum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
auth_token = os.getenv("AUTH_TOKEN")

MODEL_NAME = ModelEnum.L
LOCAL_DIR = "./models/" + ModelEnum.get_model_name(MODEL_NAME)

try:
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