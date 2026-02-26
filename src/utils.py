import yaml
import torch
import json
from transformers import CLIPProcessor, CLIPModel

def load_config(config_path="config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_clip_model(model_name):
    print("⏳ Chargement de CLIP...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    return model, processor, device

def save_paths(paths, filepath):
    with open(filepath, 'w', encoding="utf-8") as f:
        json.dump(paths, f)

def load_paths(filepath):
    with open(filepath, 'r', encoding="utf-8") as f:
        return json.load(f)