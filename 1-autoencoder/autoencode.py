import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns
import joblib
import os

from tensorflow.keras import layers, Model

from autoencoder_model import create_model, save_history_plot
from autoencoder_load_images import load_liste_images

from dotenv import load_dotenv
from pathlib import Path

load_dotenv()
image_path = Path(os.getenv("PATH_DATASET"))

# output_path : parent directory et output
output_path = Path(__file__).parent.parent.joinpath("output")

""" Ce script charge les images, crée un autoencodeur, l'entraîne sur les images et sauvegarde le modèle 
et l'historique de l'entraînement."""
help = """Usage : python autoencode.py [--no_train]
Arguments : 
- --no_train : si présent, ne pas entraîner le modèle, juste le charger depuis le fichier autoencoder.joblib
"""

### TRAITEMENT DES ARGUMENTS ###
import sys
no_train = False
if len(sys.argv) > 1 and sys.argv[1] == "--no_train":
    no_train = True
    print("Mode : pas d'entraînement, chargement du modèle depuis le fichier autoencoder.joblib")
# si -h ou --help est présent, afficher l'aide
if len(sys.argv) > 1 and sys.argv[1] in ["-h", "--help"]:
    print(help)
    sys.exit(0)
# si d'autres arguments sont présents, afficher une erreur
if len(sys.argv) > 2:
    print("Erreur : arguments non reconnus :", sys.argv[2:])
    print("Usage : python autoencode.py [no_train]")
    sys.exit(1)

### SCRIPT PRINCIPAL ###
resized_dimension = (64,64)

# Si le répertoire output n'existe pas, on le crée
if not os.path.exists(output_path):
    os.makedirs(output_path)

images_originales, images = load_liste_images(image_path, resized_dimension, category='bottle', type='train')

print(f"Nombre d'images chargées : {len(images)}")
# encodeur
if no_train:
    autoencoder = joblib.load(output_path / "autoencoder.joblib")
    encoder = joblib.load(output_path / "encoder.joblib")
    decoder = joblib.load(output_path / "decoder.joblib")
else:
    encoder, decoder, autoencoder = create_model(resized_dimension)

    autoencoder.summary()

    history = autoencoder.fit(
        images, images, 
        batch_size=32, 
        epochs=40, 
        shuffle=True,
        validation_split=0.15,  # 15% des images pour la validation
        verbose=1, 
    )

    # Sauvegarde du modèle
    joblib.dump(autoencoder, output_path / "autoencoder.joblib")
    joblib.dump(encoder, output_path / "encoder.joblib")
    joblib.dump(decoder, output_path / "decoder.joblib")

    save_history_plot(history, output_path / "history_plot.png")

# Visualisation des images reconstruites
nb_col = 6
test_pred = autoencoder.predict(images[:nb_col])

plt.figure(figsize=(14,8))
for i, image_originale in enumerate(images_originales[:nb_col]):
    
    image_autoencodee = test_pred[i].reshape(resized_dimension[0], resized_dimension[1]) * 255.
    image_erreur = np.abs(image_originale - image_autoencodee)
    
    plt.subplot(3,nb_col, i+1)
    plt.imshow( image_originale , cmap="gray" )
    plt.axis('off')
    plt.title("Original")
    
    plt.subplot(3,nb_col, i+1+nb_col)
    plt.imshow( image_autoencodee , cmap="gray" )
    plt.axis('off')
    plt.title("Auto-encodé")
    
    plt.subplot(3,nb_col, i+1+nb_col*2)
    plt.imshow( image_erreur , cmap="hot" )
    plt.axis('off')
    plt.title(f"Erreur\nMAE={image_erreur.mean():.2f}\n(max={image_erreur.max() :.0f})")
    
plt.savefig(output_path / "images_reconstruites.png")