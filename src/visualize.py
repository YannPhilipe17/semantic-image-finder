import matplotlib.pyplot as plt
from PIL import Image

def show_result(image_path, score, query):
    """Affiche l'image trouvée avec son score."""
    img = Image.open(image_path)
    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    plt.title(f"Requête : '{query}'\nScore : {score*100:.2f}%")
    plt.axis('off')
    plt.show()