import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras import layers, Model

def create_model(resized_dimension=(128,128)):
    # encodeur
    encoder_input = layers.Input(
        shape=(resized_dimension[0]*resized_dimension[1],), 
        name="input"
    )
    x = layers.Dense(
        256, 
        activation="relu", 
        name="enc_dense1"
    )(encoder_input)
    latent = layers.Dense(
        32, 
        activation="relu", 
        name="bottleneck"
    )(x)

    encoder = Model(encoder_input, latent, name="encodeur")

    # décodeur
    decoder_input = layers.Input(
        shape=(32,), 
        name="latent_input"
    )
    x = layers.Dense(
        256, 
        activation="relu", 
        name="dec_dense1"
    )(decoder_input)
    decoder_output = layers.Dense(
        resized_dimension[0]*resized_dimension[1], 
        activation="sigmoid", 
        name="output"
    )(x)

    decoder=Model(decoder_input, decoder_output, name="decodeur")

    # auto-encodeur
    autoencoder_output = decoder(encoder(encoder_input))
    autoencoder = Model(encoder_input, autoencoder_output, name="auto_encodeur")

    autoencoder.compile(
        optimizer="adam",
        loss="mse",
    )
    return encoder, decoder, autoencoder

def save_history_plot(history, file_name):
    """
    Visualisation de l'historique d'entraînement
    - history : objet History retourné par la méthode fit() du modèle
    - file_name : nom du fichier de sortie pour sauvegarder le graphique
    """
    plt.figure(figsize=(16,6))
    plt.plot(history.history["loss"], "--", label="Entraînement")
    plt.plot(history.history["val_loss"], "-", label="Validation")
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.savefig(file_name)
