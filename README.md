# 🔍 Semantic Image Finder (Powered by OpenAI CLIP)

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-orange)

##  À propos
Ce projet est un **moteur de recherche d'images sémantique local**. Il vous permet d'utiliser le langage naturel (ex: *"un chat qui dort sur un clavier"*) pour retrouver instantanément une image spécifique dans vos dossiers; ce qui signifit : Fini la recherche fastidieuse par noms de fichiers ou par dates ! 

Il utilise le modèle **CLIP** (Contrastive Language-Image Pre-training) d'OpenAI pour lier la compréhension du texte et de l'image.

##  Fonctionnalités
- **Recherche Sémantique :** Comprend le contexte et le contenu de l'image, pas juste les métadonnées.
- **100% Local & Privé :** Le modèle tourne sur votre machine. Vos photos personnelles restent chez vous.
- **Recherche Instantanée :** Pré-calcul des *embeddings* (vecteurs) pour une recherche en quelques millisecondes.

##  Installation

1. Clonez ce dépôt (ou téléchargez-le) :
```bash
git clone https://github.com/YannPhilipe17/semantic-image-finder.git
cd semantic-image-finder
  ```
2. Créer l'environnement virtuel
   ```bash
python -m venv venv

# Activer l'environnement (Windows)
.\venv\Scripts\activate

# Activer l'environnement (Mac/Linux)
source venv/bin/activate
```

##  Installation & Configuration

### 1. Cloner le projet
```bash
git clone [https://github.com/YannPhilipe17/semantic-image-finder.git](https://github.com/YannPhilipe17/semantic-image-finder.git)
cd semantic-image-finder
```

### 2. Créer l'environnement virtuel
Il est fortement recommandé d'utiliser un environnement isolé pour éviter les conflits de versions.

```bash
python -m venv venv

# Activer l'environnement (Windows)
.\venv\Scripts\activate

# Activer l'environnement (Mac/Linux)
source venv/bin/activate
```

### 3. Installer les dépendances
```bash
pip install -r requirements.txt
```

## Utilisation
Le projet fonctionne en deux étapes : l'indexation (analyse des images) et la recherche.

**Étape 1 : Indexation**
Placez vos photos dans le dossier data/images/, puis lancez l'analyse pour générer les empreintes mathématiques (embeddings) :

```bash
python main.py --index
```

**Étape 2 : Recherche**
Lancez une recherche en langage naturel (français ou anglais) :

```bash
python main.py --search "Une description de votre image"
```

### Structure 
```bash
semantic-image-finder/
├── data/images/          # Vos photos à indexer
├── embeddings/           # Vecteurs générés 
├── src/                  # Code source (logique, recherche, visualisation)
├── config.yaml           # Configuration du modèle et des chemins
└── main.py               # Point d'entrée CLI
```
