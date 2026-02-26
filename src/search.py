import torch
import numpy as np
from src.similarity import calculate_similarity
from src.utils import load_paths
from deep_translator import GoogleTranslator 

def search_image(query, embeddings_file, paths_file, model, processor, device):
    try:
        image_features_np = np.load(embeddings_file)
        image_paths = load_paths(paths_file)
    except FileNotFoundError:
        print("❌ Fichiers d'embeddings introuvables. Veuillez d'abord indexer les images avec python main.py --index.")
        return None, None

    image_features = torch.tensor(image_features_np).to(device)

    query_en = GoogleTranslator(source='auto', target='en').translate(query)
    print(f"Traduction : '{query}' -> '{query_en}'")
    # ----------------------------------------

    # On donne la version anglaise au modèle (car entraîné principalement sur de l'anglais)
    inputs = processor(text=[query_en], return_tensors="pt", padding=True).to(device)
    
    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
        
        if not isinstance(text_features, torch.Tensor):
            if hasattr(text_features, 'text_embeds'):
                text_features = text_features.text_embeds
            elif hasattr(text_features, 'pooler_output'):
                text_features = text_features.pooler_output
            else:
                text_features = text_features[0]

    similarities = calculate_similarity(text_features, image_features)
    
    best_idx = similarities.argmax().item()
    best_score = similarities[best_idx].item()
    best_path = image_paths[best_idx]
    
    return best_path, best_score