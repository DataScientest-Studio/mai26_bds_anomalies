import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns
import joblib
import os

import autoencode_figures as fig

import tensorflow as tf

from keras.utils import image_dataset_from_directory

from autoencoder_model import create_model, save_history_plot, get_callbacks, load_autoencoder, calculate_errors_labels
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
    x_modified = data_augmenter.augment(x)
    return (x_modified, x_modified)

### SCRIPT PRINCIPAL ###

# Si le répertoire output n'existe pas, on le crée
if not os.path.exists(output_path):
    os.makedirs(output_path)


##########################################
####          SETTINGS                ####
##########################################

#categories = ['bottle', 'cable', 'capsule', 'carpet', 'grid',
#    'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
#    'tile', 'toothbrush', 'transistor', 'wood', 'zipper',
#    'metal_plate']
categories = ['bottle', 'transistor']

resized_dimension = (64,64)
batch_size = 32

color_augmentation=False
move_augmentation=False

model_type = 'conv_dense' # 'conv', 'dense_conv', 'conv_dense', 'dense'
loss = 'mae' # 'mae', 'mse'
error_score = 'mse' # 'mae', 'mse'

threshold_percentile = 80

##########################################

data_augmenter = DataAugmentation(colors=False, moves=False)

for category in categories:

    # Save model parameters
    with open(output_path / f"{category}_parameters.txt", "w") as f:
        f.write(f"--- Paramètres utilisés - catégorie {category} ---\n")
        f.write(f" resized_dimension = {str(resized_dimension)}\n")
        f.write(f"        batch_size = {str(batch_size)}\n")
        f.write(f"color_augmentation = {str(color_augmentation)}\n")
        f.write(f" move_augmentation = {str(move_augmentation)}\n")
        f.write(f"        model_type = {str(model_type)}\n")
        f.write(f"              loss = {str(loss)}\n")
        f.write(f"       error_score = {str(error_score)}\n")

    model_file = output_path / f"{category}_autoencoder.keras"

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
        autoencoder = create_model(model=model_type, loss=loss, error_score=error_score, 
                                   resized_dimension= resized_dimension, nb_channels= nb_channels)

        autoencoder.summary()

        history = autoencoder.fit(
            train_ds, validation_data=val_ds, 
            epochs=100, 
            verbose=1, shuffle=False, 
            callbacks=get_callbacks(model_file, error_score=error_score), 
        )

        save_history_plot(history, output_path / f"{category}_history_plot.png")

    autoencoder = load_autoencoder(model_file)

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
    train_errors, _ = calculate_errors_labels(autoencoder, trainf_ds, error_score=error_score)
    error_threshold = np.percentile(train_errors, threshold_percentile)
    print(f"{threshold_percentile}ème percentile train =", error_threshold)

    val_errors, _ = calculate_errors_labels(autoencoder, val_ds, error_score=error_score)
    error_threshold = np.percentile(val_errors, threshold_percentile)
    print("Threshold Error (val) =", error_threshold)

    # Calcul des MSEs sur le test (avec le label 0 si good, 1 si anomalie)
    test_errors, y_true, y_pred = calculate_errors_labels(autoencoder, test_ds, error_score=error_score, errors_threshold=error_threshold)

    ##################
    # Visualisations #
    ##################

    # Visualisation des images originales reconstruites
    fig.compare_orig_encoded(trainf_ds, autoencoder, output_path, f"{category}_images_reconstruites_train_good.png")

    # Visualisation des images augmentées reconstruites
    if color_augmentation or move_augmentation:
        fig.compare_orig_encoded(train_ds, autoencoder, output_path, f"{category}_images_reconstruites_train_augmented.png")

    # Histogramme des erreurs sur les images d'entraînement (bonnes)

    fig.histogramme_erreurs(
        train_errors, 
        test_errors, y_true, 
        error_threshold, 
        output_path, 
        f"{category}_histogramme_erreurs.png", 
        category
    )

    # matrice de confusion
    fig.draw_confusion_matrix(y_true, y_pred, output_path, f"{category}_matrice_confusion.png", category)

    # classification_report
    comment = f"{category.upper()} - seuil {threshold_percentile}% ({error_threshold :.10f})"
    fig.save_classification_report(y_true, y_pred, output_path, f"{category}_classification_report.txt", comment)

    for p in [85, 90, 95, 99]:
        threshold_mse_new = np.percentile(val_errors, p)

        # Calcul des MSEs sur le test (avec le label 0 si good, 1 si anomalie)
        y_pred_new = (test_errors > threshold_mse_new).astype(int)
        
        comment = f"{category.upper()} - seuil {p}% ({threshold_mse_new :.10f})"
        fig.save_classification_report(y_true, y_pred_new, output_path, f"{category}_classification_report.txt", comment, append=True)

    # ROC curve
    fig.draw_roc_curve(test_errors, 
                    y_true, 
                    output_path, 
                    output_filename=f"{category}_roc_curve.png", 
                    category=category)

    # Visualisation des images reconstruites pour les anomalies
    fig.compare_orig_encoded(test_ds, autoencoder, output_path, f"{category}_images_reconstruites_test_anomalies.png", only_label=1)
