import torch

def calculate_similarity(text_features, image_features):
    """Calcule la similarité cosinus entre le texte et toutes les images."""
    # Normalisation
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    
    # Produit scalaire
    similarities = (text_features @ image_features.T).squeeze(0)
    return similarities