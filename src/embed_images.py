import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from src.utils import save_paths

def create_embeddings(images_dir, embeddings_file, paths_file, model, processor, device):
    print(f"Analyse du dossier {images_dir}...")
    valid_extensions = ('.png', '.jpg', '.jpeg')
    image_paths = [os.path.join(images_dir, f) for f in os.listdir(images_dir) if f.lower().endswith(valid_extensions)]
    
    if not image_paths:
        print(" Aucune image trouvée !")
        return

    all_features = []
    valid_paths = []

    for img_path in tqdm(image_paths, desc="Génération des embeddings"):
        try:
            image = Image.open(img_path).convert("RGB")
            # On génère les inputs
            inputs = processor(images=image, return_tensors="pt").to(device)

            with torch.no_grad():
                features = model.get_image_features(pixel_values=inputs['pixel_values'])
                
                
                if not isinstance(features, torch.Tensor):
                    if hasattr(features, 'pooler_output'):
                        features = features.pooler_output
                    elif hasattr(features, 'image_embeds'):
                        features = features.image_embeds
                    else:
                        features = features[0]
                        
                all_features.append(features.cpu().numpy())
                valid_paths.append(img_path)
                
        except Exception as e:
            print(f" Erreur avec {img_path}: {e}")

    if not all_features:
        print(" Impossible de générer les embeddings.")
        return

    embeddings_matrix = np.vstack(all_features)
    os.makedirs(os.path.dirname(embeddings_file), exist_ok=True)
    
    np.save(embeddings_file, embeddings_matrix)
    save_paths(valid_paths, paths_file)
    print(f" {len(valid_paths)} embeddings sauvegardés dans {embeddings_file}")