import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns
import joblib
import os

import autoencode_figures as fig

import tensorflow as tf

from keras.utils import image_dataset_from_directory

from autoencoder_model import create_model, save_history_plot, get_callbacks, load_autoencoder, calculate_mse_labels
from autoencode_data_augment import DataAugmentation

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

AUTOTUNE = tf.data.AUTOTUNE

### Util ###
def detect_nb_couleurs(ds):
    for batch in ds.take(1):
        if isinstance(batch, tuple):
            images = batch[0]
        else:
            images = batch
        return images[0].shape[-1]
    
def augment_autoencoder(data_augmenter, x):
    x_modified = data_augmenter.normalize(x) # pas d'augmentation
    return (x_modified, x_modified)

### SCRIPT PRINCIPAL ###
resized_dimension = (256,256)
batch_size = 8

# Si le répertoire output n'existe pas, on le crée
if not os.path.exists(output_path):
    os.makedirs(output_path)

# encodeur
#categories = ['bottle', 'cable', 'capsule', 'carpet', 'grid',
#    'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
#    'tile', 'toothbrush', 'transistor', 'wood', 'zipper',
#    'metal_plate']
categories = ['capsule']

data_augmenter = DataAugmentation()

for category in categories:

    model_file = output_path / f"autoencoder_{category}.keras"

    train_ds = image_dataset_from_directory(
        image_path / category / 'train', batch_size=batch_size, 
        image_size=resized_dimension, 
        validation_split=0.15, subset="training", seed=42, 
        label_mode=None, 
    )
    train_ds = train_ds.map(lambda x: augment_autoencoder(data_augmenter, x)).prefetch(AUTOTUNE)

    val_ds = image_dataset_from_directory(
        image_path / category / 'train', batch_size=batch_size, 
        image_size=resized_dimension, 
        validation_split=0.15, subset="validation", seed=42, 
        label_mode=None, 
    )
    val_ds = val_ds.map(lambda x: augment_autoencoder(data_augmenter, x)).prefetch(AUTOTUNE)

    train_cat = (not no_train) | (not (model_file).is_file())
    if train_cat:

        nb_channels = detect_nb_couleurs(train_ds)
        autoencoder = create_model(resized_dimension, nb_channels)

        autoencoder.summary()

        history = autoencoder.fit(
            train_ds, validation_data=val_ds, 
            epochs=100, 
            verbose=1, shuffle=False, 
            callbacks=get_callbacks(model_file), 
        )

        # Sauvegarde du modèle
        #joblib.dump(autoencoder, output_path / f"autoencoder_{category}.joblib")
        #joblib.dump(encoder, output_path / "encoder.joblib")
        #joblib.dump(decoder, output_path / "decoder.joblib")

        save_history_plot(history, output_path / f"history_plot_{category}.png")

    else:
        autoencoder = load_autoencoder(model_file)

        #autoencoder = joblib.load(output_path / f"autoencoder_{category}.joblib")
        #encoder = joblib.load(output_path / "encoder.joblib")
        #decoder = joblib.load(output_path / "decoder.joblib")

    #########################
    # Chargement des images #
    #########################

    # Train full (pas de split validation)
    trainf_ds = image_dataset_from_directory(
        image_path / category / 'train', batch_size=batch_size, 
        image_size=resized_dimension, 
    )
    trainf_ds = trainf_ds.map(
        lambda x, y: (
            data_augmenter.normalize(x), #tf.cast(x, tf.float32) / 255.0, 
            tf.zeros_like(y, dtype=tf.int32) 
        )
    ).prefetch(AUTOTUNE)

    test_ds = image_dataset_from_directory(
        image_path / category / 'test', batch_size=batch_size, 
        image_size=resized_dimension, 
    )
    good_value = test_ds.class_names.index('good')
    test_ds = test_ds.map(
        lambda x, y: (
            data_augmenter.normalize(x), 
            tf.cast(y != good_value, tf.int32) 
        )
    ).prefetch(AUTOTUNE)

    # Calcul des MSEs sur le train et du threshold
    train_mses, _ = calculate_mse_labels(autoencoder, trainf_ds)
    threshold_mse = np.percentile(train_mses, 95)
    print("95ème percentile train =", threshold_mse)

    val_mses, _ = calculate_mse_labels(autoencoder, val_ds)
    threshold_mse = np.percentile(val_mses, 95)
    print("Threshold MSE (val) =", threshold_mse)

    # Calcul des MSEs sur le test (avec le label 0 si good, 1 si anomalie)
    test_mses, y_true, y_pred = calculate_mse_labels(autoencoder, test_ds, threshold_mse)

    ##################
    # Visualisations #
    ##################

    # Visualisation des images originales reconstruites
    fig.compare_orig_encoded(trainf_ds, autoencoder, output_path, f"images_reconstruites_train_good_{category}.png")

    # Visualisation des images augmentées reconstruites
    fig.compare_orig_encoded(train_ds, autoencoder, output_path, f"images_reconstruites_train_augmented_{category}.png")

    # Histogramme des erreurs sur les images d'entraînement (bonnes)

    fig.histogramme_erreurs(
        train_mses, 
        test_mses, y_true, 
        threshold_mse, 
        output_path, 
        f"histogramme_erreurs_{category}.png", 
        category
    )

    # matrice de confusion
    fig.draw_confusion_matrix(y_true, y_pred, output_path, f"matrice_confusion_{category}.png", category)

    # classification_report
    comment = f"{category.upper()} - seuil 95% ({threshold_mse :.10f})"
    fig.save_classification_report(y_true, y_pred, output_path, f"classification_report_{category}.txt", comment)

    for p in [97, 98, 99, 99.5]:
        threshold_mse_new = np.percentile(val_mses, p)

        # Calcul des MSEs sur le test (avec le label 0 si good, 1 si anomalie)
        y_pred_new = (test_mses > threshold_mse_new).astype(int)
        
        comment = f"{category.upper()} - seuil {p}% ({threshold_mse_new :.10f})"
        fig.save_classification_report(y_true, y_pred_new, output_path, f"classification_report_{category}.txt", comment, append=True)

    # ROC curve
    fig.draw_roc_curve(test_mses, 
                    y_true, 
                    output_path, 
                    output_filename=f"roc_curve_{category}.png", 
                    category=category)

    # Visualisation des images reconstruites pour les anomalies
    fig.compare_orig_encoded(test_ds, autoencoder, output_path, f"images_reconstruites_test_anomalies_{category}.png", only_label=1)
