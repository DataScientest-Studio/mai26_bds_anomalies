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
output_path = Path(__file__).parent.parent.joinpath("output", Path(__file__).parent.stem)

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

# encodeur
category = 'bottle'
if no_train:
    autoencoder = joblib.load(output_path / f"autoencoder_{category}.joblib")
    #encoder = joblib.load(output_path / "encoder.joblib")
    #decoder = joblib.load(output_path / "decoder.joblib")

    images = load_liste_images(image_path, resized_dimension, category=category, type='train', quality="good", include_augmented=False)
    print(f"Nombre d'images chargées : {len(images)}")
else:
    images = load_liste_images(image_path, resized_dimension, category=category, type='train', quality="good", include_augmented=True)
    print(f"Nombre d'images chargées : {len(images)}")

    encoder, decoder, autoencoder = create_model(resized_dimension)

    autoencoder.summary()

    history = autoencoder.fit(
        images, images, 
        batch_size=32, 
        epochs=30, 
        shuffle=True,
        validation_split=0.1,  # 10% des images pour la validation
        verbose=1, 
    )

    # Sauvegarde du modèle
    joblib.dump(autoencoder, output_path / f"autoencoder_{category}.joblib")
    #joblib.dump(encoder, output_path / "encoder.joblib")
    #joblib.dump(decoder, output_path / "decoder.joblib")

    save_history_plot(history, output_path / "history_plot.png")

# Visualisation des images reconstruites
nb_col = 6
test_pred = autoencoder.predict(images[:nb_col])

plt.figure(figsize=(14,8))
for i in range(nb_col):
    
    image_originale = images[i].reshape(resized_dimension[0], resized_dimension[1])
    image_autoencodee = test_pred[i].reshape(resized_dimension[0], resized_dimension[1])
    image_erreur = np.abs(image_originale - image_autoencodee)
    
    plt.subplot(3,nb_col, i+1)
    plt.imshow( image_originale, cmap="gray" , vmin=0, vmax=1)
    plt.axis('off')
    plt.title("Original")
    
    plt.subplot(3,nb_col, i+1+nb_col)
    plt.imshow( image_autoencodee , cmap="gray" , vmin=0, vmax=1)
    plt.axis('off')
    plt.title("Auto-encodé")
    
    plt.subplot(3,nb_col, i+1+nb_col*2)
    plt.imshow( image_erreur , cmap="hot" , vmin=0, vmax=1)
    plt.axis('off')
    mae = np.mean(np.abs(image_originale - image_autoencodee))
    mse = np.mean((image_originale - image_autoencodee) ** 2)

    plt.title(f"Erreur\nMAE={mae:.5f}\nMSE={mse:.5f}")
    
plt.savefig(output_path / "images_reconstruites_train_good.png")

# Histogramme des erreurs sur les images d'entraînement (bonnes)
test_pred_good = autoencoder.predict(images)
mse_train_good = ((images - test_pred_good)**2).mean(axis=1)

plt.figure(figsize=(10,6))
plt.hist(mse_train_good, bins=30, color='green', alpha=0.7, label='Train (bonnes)')
plt.xlabel('Erreur')
plt.ylabel('Fréquence')
plt.title(f'Histogramme des erreurs - {category}')

# Définition d'une limite pour détecter les anomalies (par exemple, le 95ème percentile des erreurs sur les images d'entraînement)
threshold = np.percentile(mse_train_good, 95)
plt.axvline(threshold, color='red', linestyle='dashed', linewidth=2, label=f'Seuil (95ème percentile) = {threshold:.5f}')

# Histogramme des erreurs sur les images de test (anomalies)
images_test_anomaly = load_liste_images(image_path, resized_dimension, category=category, type='test', quality="anomaly", include_augmented=False)
print(f"Nombre d'images de test chargées : {len(images_test_anomaly)}")

test_pred_anomaly = autoencoder.predict(images_test_anomaly)
mse_anomaly = ((images_test_anomaly - test_pred_anomaly)**2).mean(axis=1)

plt.hist(mse_anomaly, bins=30, color='orange', alpha=0.7, label='Test (anomalies)')
plt.legend()
plt.savefig(output_path / f"histogramme_erreurs_{category}.png")

# Détection des anomalies sur les images de test
images_test_good = load_liste_images(image_path, resized_dimension, category=category, type='test', quality="good", include_augmented=False)
print(f"Nombre d'images de test chargées : {len(images_test_good)}")

test_pred_good = autoencoder.predict(images_test_good)
mse_test_good = ((images_test_good - test_pred_good)**2).mean(axis=1)

# matrice de confusion
from sklearn.metrics import confusion_matrix
import seaborn as sns

y_true = np.concatenate([np.zeros(len(mse_test_good)), np.ones(len(mse_anomaly))])
y_pred = np.concatenate([mse_test_good > threshold, mse_anomaly > threshold])

plt.figure(figsize=(8,6))
cm = confusion_matrix(y_true, y_pred, normalize='true')
sns.heatmap(cm, annot=True, fmt='.3f', cmap='Blues', xticklabels=['Good', 'Anomalies'], yticklabels=['Good', 'Anomalies'])
plt.xlabel('Prédiction')
plt.ylabel('Réel')
plt.title(f'Matrice de confusion - {category}')
plt.savefig(output_path / f"matrice_confusion_{category}.png")

# ROC curve
from sklearn.metrics import roc_curve, auc

fpr, tpr, thresholds = roc_curve(np.concatenate([np.zeros(len(mse_train_good)), np.ones(len(mse_anomaly))]), np.concatenate([mse_train_good, mse_anomaly]))
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.yticks(np.arange(0, 1.06, 0.05))
plt.grid()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.savefig(output_path / f"roc_curve.png")

# Visualisation des images reconstruites pour les anomalies
# Visualisation des images reconstruites
nb_col = 6
test_pred = autoencoder.predict(images_test_anomaly[:nb_col])

plt.figure(figsize=(14,8))
for i in range(nb_col):
    
    image_originale = images_test_anomaly[i].reshape(resized_dimension[0], resized_dimension[1])
    image_autoencodee = test_pred[i].reshape(resized_dimension[0], resized_dimension[1])
    image_erreur = np.abs(image_originale - image_autoencodee)
    
    plt.subplot(3,nb_col, i+1)
    plt.imshow( image_originale, cmap="gray" , vmin=0, vmax=1)
    plt.axis('off')
    plt.title("Original")
    
    plt.subplot(3,nb_col, i+1+nb_col)
    plt.imshow( image_autoencodee , cmap="gray" , vmin=0, vmax=1)
    plt.axis('off')
    plt.title("Auto-encodé")
    
    plt.subplot(3,nb_col, i+1+nb_col*2)
    plt.imshow( image_erreur , cmap="hot" , vmin=0, vmax=1)
    plt.axis('off')
    mae = np.mean(np.abs(image_originale - image_autoencodee))
    mse = np.mean((image_originale - image_autoencodee) ** 2)

    plt.title(f"Erreur\nMAE={mae:.5f}\nMSE={mse:.5f}")
    
plt.savefig(output_path / "images_reconstruites_test_anomalies.png")