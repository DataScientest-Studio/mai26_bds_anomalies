import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Rescaling, Conv2D, MaxPooling2D, UpSampling2D, Dropout
from tensorflow.keras.layers import RandomRotation, RandomZoom, RandomContrast, RandomBrightness, RandomTranslation

import tensorflow_probability as tfp

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

def load_autoencoder(filepath):
    return load_model(filepath)

def get_callbacks(filepath):
    callbacks=[]

    callbacks.append(
        ReduceLROnPlateau(monitor="val_loss", mode='min', min_delta=0.001, patience=8, cooldown=3, factor=0.5)
    )
    callbacks.append(
        EarlyStopping(monitor="val_mae", mode='min', min_delta=0.0001, patience=15, restore_best_weights=True)
    )
    callbacks.append(
        ModelCheckpoint(filepath=filepath, monitor="val_loss", mode='min', save_best_only=True)
    )

    return callbacks

def create_model(resized_dimension=(256,256), nb_channels=3):

    # encodeur
    inputs = Input(
        shape=(resized_dimension[0], resized_dimension[1], nb_channels), 
        name="input"
    )
    x=inputs

    x = Conv2D(
        16,
        kernel_size=(3,3), 
        #strides=(2,2),
        padding='same', 
        activation="relu",
    )(x)
    x = MaxPooling2D(
        (2,2), 
        padding='same', 
    )(x)
    # x = Dropout( 0.2 )(x)
    
    x = Conv2D(
        32,
        kernel_size=(3,3), 
        #strides=(2,2),
        padding='same', 
        activation="relu",
    )(x)
    x = MaxPooling2D(
        (2,2), 
        padding='same', 
    )(x)
    # x = Dropout( 0.2 )(x)
    
    x = Conv2D(
        64,
        kernel_size=(3,3), 
        #strides=(2,2),
        padding='same', 
        activation="relu",
    )(x)
    x = MaxPooling2D(
        (2,2), 
        padding='same', 
    )(x)

    # décodeur
    x = Conv2D(
        64,
        kernel_size=(3,3), 
        padding='same', 
        activation="relu",
    )(x)
    x = UpSampling2D(
        (2,2), 
    )(x)
    # x = Dropout( 0.2 )(x)

    x = Conv2D(
        32,
        kernel_size=(3,3), 
        padding='same', 
        activation="relu",
    )(x)
    x = UpSampling2D(
        (2,2), 
    )(x)
    # x = Dropout( 0.2 )(x)

    x = Conv2D(
        16,
        kernel_size=(3,3), 
        padding='same', 
        activation="relu",
    )(x)
    x = UpSampling2D(
        (2,2), 
    )(x)
    
    outputs = Conv2D(
        nb_channels,
        kernel_size=(3,3), 
        padding='same', 
        activation="sigmoid",
    )(x)

    # auto-encodeur
    autoencoder = Model(inputs = inputs, outputs = outputs, name="auto_encodeur")

    autoencoder.compile(
        optimizer="adam",
        loss="mae",
        metrics=["mse"],
    )
    return autoencoder

def save_history_plot(history, file_name):
    """
    Visualisation de l'historique d'entraînement
    - history : objet History retourné par la méthode fit() du modèle
    - file_name : nom du fichier de sortie pour sauvegarder le graphique
    """
    plt.figure(figsize=(16,6))
    plt.plot(history.history["loss"], "--", label="Entraînement")
    if "val_loss" in history.history:
        plt.plot(history.history["val_loss"], "-", label="Validation")
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.savefig(file_name)

def calculate_local_error(batch_images, pred):
    error_map = tf.reduce_mean(
        tf.square(batch_images - pred),
        axis=-1
    )
    flat_errors = tf.reshape(error_map, (tf.shape(error_map)[0], -1))

    score_p99 = tfp.stats.percentile(flat_errors, 99, axis=1)

    return score_p99

def calculate_mse_labels(model, ds, mse_threshold=None):

    mses = []
    true_labels = []
    pred_labels = []
    for batch_images, batch_labels in ds:
        pred = model(batch_images, training=False)
        
        # Calcul MSE
        batch_mses = tf.reduce_mean(tf.square(batch_images - pred), axis=(1,2,3))

        # Calcul MAE
        # batch_mses = tf.reduce_mean(tf.abs(batch_images - pred), axis=(1,2,3))
        
        # Calcul erreur locale
        # batch_mses = calculate_local_error(batch_images, pred)

        mses.extend(batch_mses.numpy())
        
        if mse_threshold is not None:
            pred_labels.extend( (batch_mses > mse_threshold).numpy().astype(int) )

        true_labels.extend(batch_labels.numpy())

    if mse_threshold is None:
        return mses, true_labels
    else:
        return mses, true_labels, pred_labels