import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras import layers, Model

@tf.keras.utils.register_keras_serializable()
class Sampling(layers.Layer):
    """Echantillonne z à partir de (z_mean, z_log_var) via le reparameterization trick."""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


@tf.keras.utils.register_keras_serializable()
class KLLossLayer(layers.Layer):
    """Calcule la perte KL à partir de (z_mean, z_log_var) et l'ajoute au modèle via add_loss."""
    def __init__(self, kl_weight=1.0, **kwargs):
        super().__init__(**kwargs)
        self.kl_weight = kl_weight

    def call(self, inputs):
        z_mean, z_log_var = inputs
        kl_loss = -0.5 * tf.reduce_mean(
            tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
        )
        self.add_loss(self.kl_weight * kl_loss)
        return z_mean

    def get_config(self):
        config = super().get_config()
        config.update({"kl_weight": self.kl_weight})
        return config


def create_model(resized_dimension=(128,128), nb_channels=1, kl_weight=0.1):
    latent_space = 2**(3+nb_channels)  # 16 (gris) ou 32 (couleur)
    h, w = resized_dimension

    # ---------- ENCODEUR (convolutif) ----------
    encoder_input = layers.Input(shape=(h, w, nb_channels), name="input")

    x = layers.Conv2D(16, 3, strides=2, activation="relu", padding="same", name="enc_conv1")(encoder_input)  # h/2
    x = layers.Conv2D(32, 3, strides=2, activation="relu", padding="same", name="enc_conv2")(x)               # h/4
    x = layers.Conv2D(64, 3, strides=2, activation="relu", padding="same", name="enc_conv3")(x)               # h/8

    shape_before_flatten = x.shape[1:]  # (h/8, w/8, 64), utile pour le décodeur
    x = layers.Flatten(name="enc_flatten")(x)

    z_mean = layers.Dense(latent_space, activation="linear", name="z_mean")(x)
    z_log_var = layers.Dense(latent_space, activation="linear", name="z_log_var")(x)
    z = Sampling(name="bottleneck")([z_mean, z_log_var])

    KLLossLayer(kl_weight=kl_weight, name="kl_loss_layer")([z_mean, z_log_var])

    encoder = Model(encoder_input, [z_mean, z_log_var, z], name="encodeur")

    # ---------- DECODEUR (convolutif, symétrique) ----------
    decoder_input = layers.Input(shape=(latent_space,), name="latent_input")

    x = layers.Dense(
        shape_before_flatten[0] * shape_before_flatten[1] * shape_before_flatten[2], 
        activation="relu", 
        name="dec_dense"
    )(decoder_input)
    x = layers.Reshape(shape_before_flatten, name="dec_reshape")(x)

    x = layers.Conv2DTranspose(64, 3, strides=2, activation="relu", padding="same", name="dec_convT1")(x)   # h/4
    x = layers.Conv2DTranspose(32, 3, strides=2, activation="relu", padding="same", name="dec_convT2")(x)   # h/2
    x = layers.Conv2DTranspose(16, 3, strides=2, activation="relu", padding="same", name="dec_convT3")(x)   # h

    decoder_output = layers.Conv2D(
        nb_channels, 3, activation="sigmoid", padding="same", name="output"
    )(x)

    decoder = Model(decoder_input, decoder_output, name="decodeur")

    # ---------- VAE complet ----------
    z_mean_out, z_log_var_out, z_out = encoder(encoder_input)
    autoencoder_output = decoder(z_out)
    autoencoder = Model(encoder_input, autoencoder_output, name="auto_encodeur")

    autoencoder.compile(
        optimizer="adam",
        loss="mse",
        metrics=["mae"]
    )
    return encoder, decoder, autoencoder


def save_history_plot(history, file_name):
    """
    Visualisation de l'historique d'entraînement
    """
    plt.figure(figsize=(16,6))
    plt.plot(history.history["loss"], "--", label="Entraînement")
    if "val_loss" in history.history:
        plt.plot(history.history["val_loss"], "-", label="Validation")
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.savefig(file_name)