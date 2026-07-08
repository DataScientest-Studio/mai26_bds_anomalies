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


def create_model(resized_dimension=(128,128), nb_channels=1, kl_weight=1.0):
    latent_space = 2**(3+nb_channels)  # réduit : ex. 16 pour nb_channels=1, 32 pour nb_channels=3

    # encodeur (couches beaucoup plus petites)
    encoder_input = layers.Input(
        shape=(resized_dimension[0]*resized_dimension[1]*nb_channels,), 
        name="input"
    )
    x = layers.Dense(
        128, 
        activation="relu", 
        name="enc_dense1"
    )(encoder_input)
    x = layers.Dense(
        64, 
        activation="relu", 
        name="enc_dense2"
    )(x)

    z_mean = layers.Dense(latent_space, activation="linear", name="z_mean")(x)
    z_log_var = layers.Dense(latent_space, activation="linear", name="z_log_var")(x)
    z = Sampling(name="bottleneck")([z_mean, z_log_var])

    KLLossLayer(kl_weight=kl_weight, name="kl_loss_layer")([z_mean, z_log_var])

    encoder = Model(encoder_input, [z_mean, z_log_var, z], name="encodeur")

    # décodeur (symétrique, plus petit aussi)
    decoder_input = layers.Input(
        shape=(latent_space,), 
        name="latent_input"
    )
    x = layers.Dense(
        64, 
        activation="relu", 
        name="dec_dense1"
    )(decoder_input)
    x = layers.Dense(
        128, 
        activation="relu", 
        name="dec_dense2"
    )(x)
    decoder_output = layers.Dense(
        resized_dimension[0]*resized_dimension[1]*nb_channels, 
        activation="sigmoid", 
        name="output"
    )(x)

    decoder = Model(decoder_input, decoder_output, name="decodeur")

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