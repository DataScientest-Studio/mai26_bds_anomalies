import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns
import joblib
import os

import vae_figures as fig

from tensorflow.keras import layers, Model

from vae_model import create_model, save_history_plot
from vae_loead_images_RAD import load_liste_images

from dotenv import load_dotenv
from pathlib import Path

load_dotenv('sample.env')
image_path = Path(os.getenv("PATH_DATASET_RAD"))

output_path = Path(__file__).parent.parent.joinpath("output", Path(__file__).parent.stem)

""" Ce script charge les images, crée un VAE, l'entraîne sur les images et sauvegarde le modèle 
et l'historique de l'entraînement."""
help = """Usage : python vae.py [--no_train]
Arguments : 
- --no_train : si présent, ne pas entraîner le modèle, juste le charger depuis le fichier .joblib
"""


import sys
no_train = False
if len(sys.argv) > 1 and sys.argv[1] == "--no_train":
    no_train = True
    print("Mode : pas d'entraînement, chargement du modèle depuis le fichier .joblib")
if len(sys.argv) > 1 and sys.argv[1] in ["-h", "--help"]:
    print(help)
    sys.exit(0)
if len(sys.argv) > 2:
    print("Erreur : arguments non reconnus :", sys.argv[2:])
    print("Usage : python vae.py [--no_train]")
    sys.exit(1)

resized_dimension = (64,64)

if not os.path.exists(output_path):
    os.makedirs(output_path)


categories = ['bolt']

for category in categories:

    train_cat = (not no_train) | (not (output_path / f"autoencoder_{category}.joblib").is_file())
    if train_cat:
        
        images, nb_channels = load_liste_images(image_path, resized_dimension, category=category, type='train', quality="good", include_augmented=True)
        print(f"Nombre d'images chargées : {len(images)}")

        # reshape en 3D pour le modèle convolutif (hauteur, largeur, canaux)
        images_3d = images.reshape(-1, resized_dimension[0], resized_dimension[1], nb_channels)

        encoder, decoder, autoencoder = create_model(resized_dimension, nb_channels=nb_channels)

        autoencoder.summary()

        history = autoencoder.fit(
            images_3d, images_3d, 
            batch_size=32, 
            epochs=30, 
            shuffle=True,
            validation_split=0.1,  # 10% des images pour la validation
            verbose=1, 
        )

        # Sauvegarde du modèle
        joblib.dump(autoencoder, output_path / f"autoencoder_{category}.joblib")

        save_history_plot(history, output_path / f"history_plot_{category}.png")

    else:
        autoencoder = joblib.load(output_path / f"autoencoder_{category}.joblib")

   
    # Chargement des images 

    # Images d'entraînement
    images_train_flat, nb_channels = load_liste_images(image_path, resized_dimension, category=category, type='train', quality="good", include_augmented=False)
    print(f"Nombre d'images chargées : {len(images_train_flat)}")
    images_train = images_train_flat.reshape(-1, resized_dimension[0], resized_dimension[1], nb_channels)

    # Prédiction sur les images d'entraînement (le modèle attend et sort du 3D)
    pred_train = autoencoder.predict(images_train)
    pred_train_flat = pred_train.reshape(-1, resized_dimension[0]*resized_dimension[1]*nb_channels)

    mse_train = ((images_train_flat - pred_train_flat)**2).mean(axis=1)

    # Images de tests en anomalie
    images_test_anomaly_flat, nb_channels_test_anomaly = load_liste_images(image_path, resized_dimension, category=category, type='test', quality="anomaly", include_augmented=False)
    print(f"Nombre d'images de test chargées : {len(images_test_anomaly_flat)}")
    images_test_anomaly = images_test_anomaly_flat.reshape(-1, resized_dimension[0], resized_dimension[1], nb_channels_test_anomaly)

    pred_test_anomaly = autoencoder.predict(images_test_anomaly)
    pred_test_anomaly_flat = pred_test_anomaly.reshape(-1, resized_dimension[0]*resized_dimension[1]*nb_channels_test_anomaly)

    mse_test_anomaly = ((images_test_anomaly_flat - pred_test_anomaly_flat)**2).mean(axis=1)

    # Images de test good
    images_test_good_flat, nb_channels_test_good = load_liste_images(image_path, resized_dimension, category=category, type='test', quality="good", include_augmented=False)
    print(f"Nombre d'images de test chargées : {len(images_test_good_flat)}")
    images_test_good = images_test_good_flat.reshape(-1, resized_dimension[0], resized_dimension[1], nb_channels_test_good)

    pred_test_good = autoencoder.predict(images_test_good)
    pred_test_good_flat = pred_test_good.reshape(-1, resized_dimension[0]*resized_dimension[1]*nb_channels_test_good)

    mse_test_good = ((images_test_good_flat - pred_test_good_flat)**2).mean(axis=1)

   #visualisation

    # Visualisation des images reconstruites
    fig.compare_orig_encoded(images_train, pred_train, output_path, f"images_reconstruites_train_good_{category}.png")

    # Histogramme des erreurs sur les images d'entraînement (bonnes)
    threshold_mse = np.percentile(mse_train, 95)
    print("Threshold MSE =", threshold_mse)

    fig.histogramme_erreurs(
        images_train_flat, pred_train_flat, 
        np.concat([images_test_good_flat, images_test_anomaly_flat], axis=0), 
        np.concat([pred_test_good_flat, pred_test_anomaly_flat], axis=0), 
        np.concat([np.zeros((len(images_test_good_flat))), np.ones((len(images_test_anomaly_flat)))], axis=0), 
        threshold_mse, 
        output_path, 
        f"histogramme_erreurs_{category}.png", 
        category
    )

    # matrice de confusion
    y_true = np.concatenate([np.zeros(len(mse_test_good)), np.ones(len(mse_test_anomaly))])
    y_pred = np.concatenate([mse_test_good > threshold_mse, mse_test_anomaly > threshold_mse])

    fig.draw_confusion_matrix(y_true, y_pred, output_path, f"matrice_confusion_{category}.png", category)

    # ROC curve
    fig.draw_roc_curve(np.concatenate([mse_test_good, mse_test_anomaly]), 
                    np.concatenate([np.zeros(len(mse_test_good)), np.ones(len(mse_test_anomaly))]), 
                    output_path, 
                    output_filename=f"roc_curve_{category}.png", 
                    category=category)

    # Visualisation des images reconstruites pour les anomalies
    fig.compare_orig_encoded(images_test_anomaly, pred_test_anomaly, output_path, f"images_reconstruites_test_anomalies_{category}.png")
