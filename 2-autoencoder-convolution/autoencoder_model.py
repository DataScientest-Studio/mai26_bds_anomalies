import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Rescaling, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Dense, Flatten, Reshape
from tensorflow.keras.layers import RandomRotation, RandomZoom, RandomContrast, RandomBrightness, RandomTranslation

from tensorflow.keras.applications import EfficientNetB0

import tensorflow_probability as tfp

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

transfer_learning=None

def load_autoencoder(filepath):
    return load_model(filepath)

def get_callbacks(filepath, error_score="mae"):
    callbacks=[]

    callbacks.append(
        ReduceLROnPlateau(monitor="val_"+error_score, mode='min', min_delta=0.001, patience=8, cooldown=3, factor=0.5)
    )
    callbacks.append(
        EarlyStopping(monitor="val_"+error_score, mode='min', min_delta=0.0001, patience=15, restore_best_weights=True)
    )
    callbacks.append(
        ModelCheckpoint(filepath=filepath, monitor="val_"+error_score, mode='min', save_best_only=True)
    )

    return callbacks

def get_model_dense_conv(resized_dimension, nb_channels):

    # encodeur
    inputs = Input(
        shape=(resized_dimension[0], resized_dimension[1], nb_channels), 
        name="input"
    )
    x=inputs

    x = Flatten()(x)
    x = Dense(
        2048, 
        activation="relu", 
        name="enc_dense1"
    )(x)
    x = Dense(
        1024, 
        activation="relu", 
        name="enc_dense2"
    )(x)
    x = Dense(
        512, 
        activation="relu", 
        name="enc_dense3"
    )(x)

    x = Dense(
        768, 
        activation="relu", 
    )(x)
    x = Reshape((16,16,3))(x)

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

    return inputs, outputs

def get_model_conv(resized_dimension, nb_channels):
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

    return inputs, outputs

def get_model_conv_dense(resized_dimension, nb_channels):
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
    x = Flatten()(x)
    x = Dense(
        1024, 
        activation="relu", 
    )(x)
    x = Dense(
        (resized_dimension[0] * resized_dimension[1] * nb_channels), 
        activation="sigmoid", 
    )(x)
    outputs = Reshape(
        (resized_dimension[0], resized_dimension[1], nb_channels)
    )(x)

    return inputs, outputs

def get_model_dense(resized_dimension, nb_channels):
    # encodeur
    inputs = Input(
        shape=(*resized_dimension,nb_channels,), 
        name="input"
    )
    x = Flatten()(inputs)
    x = Dense(
        2048, 
        activation="relu", 
        name="enc_dense1"
    )(x)
    x = Dense(
        1024, 
        activation="relu", 
        name="enc_dense2"
    )(x)
    x = Dense(
        512, 
        activation="relu", 
        name="enc_dense3"
    )(x)
    latent = Dense(
        256, 
        activation="linear", 
        name="bottleneck"
    )(x)

    # décodeur
    x = Dense(
        512, 
        activation="relu", 
        name="dec_dense1"
    )(latent)
    x = Dense(
        1024, 
        activation="relu", 
        name="dec_dense2"
    )(x)
    x = Dense(
        2048, 
        activation="relu", 
        name="dec_dense3"
    )(x)
    x = Dense(
        resized_dimension[0]*resized_dimension[1]*nb_channels, 
        activation="sigmoid", 
    )(x)
    outputs = Reshape(
        (resized_dimension[0], resized_dimension[1], nb_channels)
    )(x)

    return inputs, outputs

def get_model_convtl(resized_dimension, nb_channels): 
    """Encodeur convolutionnel avec transfer learning et décodeur convolutionnel"""

    # encodeur
    inputs = Input(
        shape=(*resized_dimension, nb_channels), 
        name="input"
    )

    encoder_inputs = Rescaling(255.)(inputs)

    encoder = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(*resized_dimension, nb_channels),
    )
    # encodeur entièrement gelé.
    encoder.trainable = False

    x = encoder(encoder_inputs, training=False)

    # décodeur
    x = Conv2D(
        256,
        kernel_size=(1,1), 
        padding='same', 
        activation="relu",
    )(x)
    x = UpSampling2D(
        (2,2), 
    )(x)
    # x = Dropout( 0.2 )(x)

    x = Conv2D(
        128,
        kernel_size=(3,3), 
        padding='same', 
        activation="relu",
    )(x)
    x = UpSampling2D(
        (2,2), 
    )(x)
    # x = Dropout( 0.2 )(x)

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
    # x = Dropout( 0.2 )(x)

    x = Conv2D(
        16,
        kernel_size=(3,3), 
        padding='same', 
        activation="relu",
    )(x)
    
    outputs = Conv2D(
        nb_channels,
        kernel_size=(3,3), 
        padding='same', 
        activation="sigmoid",
    )(x)

    return inputs, outputs

def get_model_convtl_dense(resized_dimension, nb_channels): 
    """Encodeur convolutionnel avec transfer learning et décodeur dense"""

    # encodeur
    inputs = Input(
        shape=(*resized_dimension, nb_channels), 
        name="input"
    )

    encoder_inputs = Rescaling(255.)(inputs)

    encoder = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(*resized_dimension, nb_channels),
    )
    # encodeur entièrement gelé.
    encoder.trainable = False

    x = encoder(encoder_inputs, training=False)

    # décodeur
    x = Flatten()(x)
    x = Dense(
        1024, 
        activation="relu", 
    )(x)
    x = Dense(
        (resized_dimension[0] * resized_dimension[1] * nb_channels), 
        activation="sigmoid", 
    )(x)
    outputs = Reshape(
        (resized_dimension[0], resized_dimension[1], nb_channels)
    )(x)

    return inputs, outputs

def create_model(model = "conv", loss="mse", error_score="mae", resized_dimension=(256,256), nb_channels=3):

    valid_scores=['mae', 'mse']
    if loss not in valid_scores:
        raise ValueError(f"loss {loss} not valid. Must be in: '{'\', \''.join(valid_scores)}'")
    if error_score not in valid_scores:
        raise ValueError(f"error_score {error_score} not valid. Must be in: '{'\', \''.join(valid_scores)}'")

    if model=="conv":
        inputs, outputs = get_model_conv(resized_dimension, nb_channels)
    elif model=="dense_conv":
        inputs, outputs = get_model_dense_conv(resized_dimension, nb_channels)
    elif model=="conv_dense":
        inputs, outputs = get_model_conv_dense(resized_dimension, nb_channels)
    elif model=="dense":
        inputs, outputs = get_model_dense(resized_dimension, nb_channels)
    elif model=="convtl":
        inputs, outputs = get_model_convtl(resized_dimension, nb_channels)
    elif model=="convtl_dense":
        inputs, outputs = get_model_convtl_dense(resized_dimension, nb_channels)
    else:
        raise ValueError(f"model {model} not valid. Must be in: 'conv', 'dense_conv', 'conv_dense', 'dense', 'convtl', 'convtl_dense")

    # auto-encodeur
    autoencoder = Model(inputs = inputs, outputs = outputs, name="auto_encodeur")

    autoencoder.compile(
        optimizer="adam",
        loss=loss,
        metrics=[error_score],
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

def calculate_errors_labels(model, ds, error_score="mae", errors_threshold=None):

    errors = []
    true_labels = []
    pred_labels = []
    for batch_images, batch_labels in ds:
        pred = model(batch_images, training=False)
        
        # Calcul MSE
        if error_score=="mse":
            batch_errors = tf.reduce_mean(tf.square(batch_images - pred), axis=(1,2,3))

        # Calcul MAE
        elif error_score=="mae":
            batch_errors = tf.reduce_mean(tf.abs(batch_images - pred), axis=(1,2,3))
        
        else:
            raise ValueError(f"error_score {error_score} not valid. Must be in: 'mse', 'mae'")
        # Calcul erreur locale
        # batch_mses = calculate_local_error(batch_images, pred)

        errors.extend(batch_errors.numpy())
        
        if errors_threshold is not None:
            pred_labels.extend( (batch_errors > errors_threshold).numpy().astype(int) )

        true_labels.extend(batch_labels.numpy())

    if errors_threshold is None:
        return errors, true_labels
    else:
        return errors, true_labels, pred_labels