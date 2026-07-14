import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns
import joblib
import os

import autoencode_figures as fig

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
#categories = ['bottle', 'cable', 'capsule', 'carpet', 'grid',
#    'hazelnut', 'leather', 'metal_nut', 'pill', 'screw_preprocessed',
#    'tile', 'toothbrush', 'transistor', 'wood', 'zipper',
#    'metal_plate']
categories = ['screw_preprocessed']

for category in categories:

    train_cat = (not no_train) | (not (output_path / f"autoencoder_{category}.joblib").is_file())
    if train_cat:
        
        images, nb_channels = load_liste_images(image_path, resized_dimension, category=category, type='train', quality="good", include_augmented=True)
        print(f"Nombre d'images chargées : {len(images)}")

        encoder, decoder, autoencoder = create_model(resized_dimension, nb_channels=nb_channels)

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

        save_history_plot(history, output_path / f"history_plot_{category}.png")

    else:
        autoencoder = joblib.load(output_path / f"autoencoder_{category}.joblib")
        #encoder = joblib.load(output_path / "encoder.joblib")
        #decoder = joblib.load(output_path / "decoder.joblib")

    #########################
    # Chargement des images #
    #########################

    # Images d'entraînement
    images_train_flat, nb_channels = load_liste_images(image_path, resized_dimension, category=category, type='train', quality="good", include_augmented=False)
    print(f"Nombre d'images chargées : {len(images_train_flat)}")
    images_train = images_train_flat.reshape(-1, resized_dimension[0], resized_dimension[1], nb_channels)

    # Prédiction sur les images d'entraînement
    pred_train_flat = autoencoder.predict(images_train_flat)
    pred_train = pred_train_flat.reshape(-1, resized_dimension[0], resized_dimension[1], nb_channels)

    #mse_train = np.sum((images_train_flat - pred_train_flat)**2, axis=1) / len(images_train_flat)
    mse_train = ((images_train_flat - pred_train_flat)**2).mean(axis=1)

    # Images de tests en anomalie
    images_test_anomaly_flat, nb_channels_test_anomaly = load_liste_images(image_path, resized_dimension, category=category, type='test', quality="anomaly", include_augmented=False)
    print(f"Nombre d'images de test chargées : {len(images_test_anomaly_flat)}")
    images_test_anomaly = images_test_anomaly_flat.reshape(-1, resized_dimension[0], resized_dimension[1], nb_channels_test_anomaly)

    pred_test_anomaly_flat = autoencoder.predict(images_test_anomaly_flat)
    pred_test_anomaly = pred_test_anomaly_flat.reshape(-1, resized_dimension[0], resized_dimension[1], nb_channels_test_anomaly)

    mse_test_anomaly = ((images_test_anomaly_flat - pred_test_anomaly_flat)**2).mean(axis=1)

    # Images de test good
    images_test_good_flat, nb_channels_test_good = load_liste_images(image_path, resized_dimension, category=category, type='test', quality="good", include_augmented=False)
    print(f"Nombre d'images de test chargées : {len(images_test_good_flat)}")
    images_test_good = images_test_good_flat.reshape(-1, resized_dimension[0], resized_dimension[1], nb_channels_test_good)

    pred_test_good_flat = autoencoder.predict(images_test_good_flat)
    pred_test_good = pred_test_good_flat.reshape(-1, resized_dimension[0], resized_dimension[1], nb_channels_test_good)

    mse_test_good = ((images_test_good_flat - pred_test_good_flat)**2).mean(axis=1)

    ##################
    # Visualisations #
    ##################

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
    #print("y_true=", y_true[:15])
    #print("y_pred=", y_pred[:15])
    #print("mse_test_good=", mse_test_good[:15])
    #print("mse_train=", mse_train[:15])

    fig.draw_confusion_matrix(y_true, y_pred, output_path, f"matrice_confusion_{category}.png", category)

    # ROC curve
    fig.draw_roc_curve(np.concatenate([mse_test_good, mse_test_anomaly]), 
                    np.concatenate([np.zeros(len(mse_test_good)), np.ones(len(mse_test_anomaly))]), 
                    output_path, 
                    output_filename=f"roc_curve_{category}.png", 
                    category=category)

    # Visualisation des images reconstruites pour les anomalies
    fig.compare_orig_encoded(images_test_anomaly, pred_test_anomaly, output_path, f"images_reconstruites_test_anomalies_{category}.png")
