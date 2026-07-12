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


@tf.keras.utils.register_keras_serializable()
class PerceptualLossLayer(layers.Layer):
    """Compare l'image originale et la reconstruction via les features d'un VGG16 pré-entraîné
    (gelé), au lieu de comparer seulement les pixels bruts. Aide à mieux capturer les textures
    et motifs répétitifs (comme des trous de grille)."""
    def __init__(self, weight=1.0, **kwargs):
        super().__init__(**kwargs)
        self.weight = weight
        vgg = tf.keras.applications.VGG16(include_top=False, weights="imagenet")
        vgg.trainable = False
        self.feature_extractor = Model(
            vgg.input, vgg.get_layer("block2_conv2").output, name="vgg_features"
        )
        self.feature_extractor.trainable = False

    def call(self, inputs):
        y_true, y_pred = inputs

        # VGG16 attend des images en 3 canaux (RGB) : si nos images sont en niveaux de gris,
        # on duplique le canal 3 fois
        if y_true.shape[-1] == 1:
            y_true_rgb = tf.repeat(y_true, 3, axis=-1)
            y_pred_rgb = tf.repeat(y_pred, 3, axis=-1)
        else:
            y_true_rgb = y_true
            y_pred_rgb = y_pred

        #images sont en [0,1], VGG16 attend un pré-traitement spécifique sur [0,255]
        y_true_prep = tf.keras.applications.vgg16.preprocess_input(y_true_rgb * 255.0)
        y_pred_prep = tf.keras.applications.vgg16.preprocess_input(y_pred_rgb * 255.0)

        true_features = self.feature_extractor(y_true_prep)
        pred_features = self.feature_extractor(y_pred_prep)

        perceptual_loss = tf.reduce_mean(tf.square(true_features - pred_features))
        self.add_loss(self.weight * perceptual_loss)
        return y_pred

    def get_config(self):
        config = super().get_config()
        config.update({"weight": self.weight})
        return config


def create_model(resized_dimension=(128,128), nb_channels=1, kl_weight=0.1, perceptual_weight=0.5):
    latent_space = 2**(3+nb_channels)  # 16 (gris) ou 32 (couleur)
    h, w = resized_dimension

    #ENCODEUR (convolutif)
    encoder_input = layers.Input(shape=(h, w, nb_channels), name="input")

    x = layers.Conv2D(16, 3, strides=2, activation="relu", padding="same", name="enc_conv1")(encoder_input)  # h/2
    x = layers.Conv2D(32, 3, strides=2, activation="relu", padding="same", name="enc_conv2")(x)
    x = layers.Conv2D(32, 3, strides=2, activation="relu", padding="same", name="enc_conv3")(x)              # h/4

    shape_before_flatten = x.shape[1:]
    x = layers.Flatten(name="enc_flatten")(x)

    z_mean = layers.Dense(latent_space, activation="linear", name="z_mean")(x)
    z_log_var = layers.Dense(latent_space, activation="linear", name="z_log_var")(x)
    z = Sampling(name="bottleneck")([z_mean, z_log_var])

    KLLossLayer(kl_weight=kl_weight, name="kl_loss_layer")([z_mean, z_log_var])

    encoder = Model(encoder_input, [z_mean, z_log_var, z], name="encodeur")

    #DECODEUR (convolutif, symétrique)
    decoder_input = layers.Input(shape=(latent_space,), name="latent_input")

    x = layers.Dense(
        shape_before_flatten[0] * shape_before_flatten[1] * shape_before_flatten[2], 
        activation="relu", 
        name="dec_dense"
    )(decoder_input)
    x = layers.Reshape(shape_before_flatten, name="dec_reshape")(x)

    x = layers.Conv2DTranspose(32, 3, strides=2, activation="relu", padding="same", name="dec_convT1")(x)
    x = layers.Conv2DTranspose(32, 3, strides=2, activation="relu", padding="same", name="dec_convT2")(x)
    x = layers.Conv2DTranspose(16, 3, strides=2, activation="relu", padding="same", name="dec_convT3")(x)

    decoder_output = layers.Conv2D(
        nb_channels, 3, activation="sigmoid", padding="same", name="output"
    )(x)

    decoder = Model(decoder_input, decoder_output, name="decodeur")

    #VAE complet
    z_mean_out, z_log_var_out, z_out = encoder(encoder_input)
    autoencoder_output = decoder(z_out)

    # perte perceptuelle (VGG16), ajoutée en plus du MSE et de la perte KL
    PerceptualLossLayer(weight=perceptual_weight, name="perceptual_loss_layer")([encoder_input, autoencoder_output])

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
