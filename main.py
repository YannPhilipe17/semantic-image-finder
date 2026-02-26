import argparse
import os
from src.utils import load_config, load_clip_model
from src.embed_images import create_embeddings
from src.search import search_image
from src.visualize import show_result

def main():
    parser = argparse.ArgumentParser(description="Moteur de recherche sémantique d'images avec CLIP")
    parser.add_argument('--index', action='store_true', help="Indexer les images du dossier data/images")
    parser.add_argument('--search', type=str, help="Texte à rechercher dans les images")
    
    args = parser.parse_args()
    config = load_config()

    if not args.index and not args.search:
        parser.print_help()
        return

    model, processor, device = load_clip_model(config['model']['name'])

    if args.index:
        os.makedirs(config['paths']['images_dir'], exist_ok=True)
        create_embeddings(
            config['paths']['images_dir'],
            config['paths']['embeddings_file'],
            config['paths']['image_paths_file'],
            model, processor, device
        )

    if args.search:
        print(f"🔎 Recherche de : '{args.search}'")
        best_path, score = search_image(
            args.search,
            config['paths']['embeddings_file'],
            config['paths']['image_paths_file'],
            model, processor, device
        )
        
        if best_path:
            print(f"🎯 Correspondance trouvée : {best_path} (Score: {score*100:.2f}%)")
            show_result(best_path, score, args.search)

if __name__ == "__main__":
    main()