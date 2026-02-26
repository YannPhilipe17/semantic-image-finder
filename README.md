# 🔍 Semantic Image Finder (Powered by OpenAI CLIP)

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-orange)

## 📖 À propos
Ce projet est un **moteur de recherche d'images sémantique local**. Il vous permet d'utiliser le langage naturel (ex: *"un chat qui dort sur un clavier"*) pour retrouver instantanément une image spécifique dans vos dossiers; ce qui signifit : Fini la recherche fastidieuse par noms de fichiers ou par dates ! 

Il utilise le modèle **CLIP** (Contrastive Language-Image Pre-training) d'OpenAI pour lier la compréhension du texte et de l'image.

## 🚀 Fonctionnalités
- **Recherche Sémantique :** Comprend le contexte et le contenu de l'image, pas juste les métadonnées.
- **100% Local & Privé :** Le modèle tourne sur votre machine. Vos photos personnelles restent chez vous.
- **Recherche Instantanée :** Pré-calcul des *embeddings* (vecteurs) pour une recherche en quelques millisecondes.

## 🛠️ Installation

1. Clonez ce dépôt (ou téléchargez-le) :
```bash
git clone [https://github.com/YannPhilipe17/image-finder.git](https://github.com/YannPhilipe17/image-finder.git)
cd image-finder 
  ```