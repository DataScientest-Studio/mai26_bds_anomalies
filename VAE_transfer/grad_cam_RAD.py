import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import joblib
import os
from pathlib import Path
from dotenv import load_dotenv

from vae_transfer_model import create_model, save_history_plot  # nécessaire pour enregistrer Sampling/KLLossLayer avant joblib.load
from vae_transfer_load_images_RAD import load_liste_images

load_dotenv('sample.env')
image_path = Path(os.getenv("PATH_DATASET_RAD"))
output_path = Path(__file__).parent.parent.joinpath("output", Path(__file__).parent.stem)

resized_dimension = (64,64)
last_conv_layer_name = "enc_conv2"  # à adapter selon ton architecture (enc_conv2 ou enc_conv3)

categories = ['bolt', 'ribbon', 'sponge', 'tape']


def make_gradcam_heatmap(img_array, autoencoder, last_conv_layer_name, encoder_model_name="encodeur"):
    encoder = autoencoder.get_layer(encoder_model_name)
    last_conv_layer = encoder.get_layer(last_conv_layer_name)

    grad_model = tf.keras.Model(
        encoder.inputs, [last_conv_layer.output, encoder.output]
    )

    with tf.GradientTape() as tape:
        conv_output, encoder_output = grad_model(img_array)
        z_mean, z_log_var, z = encoder_output

        decoder = autoencoder.get_layer("decodeur")
        reconstruction = decoder(z_mean)  # z_mean pour un résultat déterministe (pas de bruit d'échantillonnage)

        loss = tf.reduce_mean(tf.square(img_array - reconstruction))

    grads = tape.gradient(loss, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)

    return heatmap.numpy(), reconstruction.numpy()


def overlay_heatmap(image, heatmap, alpha=0.4):
    heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    image_uint8 = np.uint8(255 * image) if image.max() <= 1 else image.astype(np.uint8)
    if image_uint8.shape[-1] == 1:
        image_uint8 = cv2.cvtColor(image_uint8, cv2.COLOR_GRAY2RGB)

    overlay = cv2.addWeighted(image_uint8, 1 - alpha, heatmap_colored, alpha, 0)

    return overlay


### SCRIPT PRINCIPAL : boucle sur toutes les catégories ###

for category in categories:

    model_file = output_path / f"autoencoder_{category}.joblib"
    if not model_file.is_file():
        print(f"Modèle non trouvé pour la catégorie '{category}' ({model_file}), skip.")
        continue

    print(f"--- Grad-CAM pour la catégorie : {category} ---")

    # Charger le modèle déjà entraîné
    autoencoder = joblib.load(model_file)

    # Charger quelques images de test en anomalie
    images_test_anomaly_flat, nb_channels = load_liste_images(
        image_path, resized_dimension, category=category, type='test', quality="anomaly", include_augmented=False
    )
    images_test_anomaly = images_test_anomaly_flat.reshape(-1, resized_dimension[0], resized_dimension[1], nb_channels)

    # Prendre la première image de test en anomalie comme exemple
    img_array = images_test_anomaly[0:1]  # shape (1, h, w, c)

    heatmap, reconstruction = make_gradcam_heatmap(img_array, autoencoder, last_conv_layer_name=last_conv_layer_name)
    overlay = overlay_heatmap(images_test_anomaly[0], heatmap)

    # Affichage : image originale, reconstruction, et heatmap superposée
    fig_plot, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(images_test_anomaly[0].squeeze(), cmap="gray" if nb_channels == 1 else None)
    axes[0].set_title("Image originale")
    axes[0].axis('off')

    axes[1].imshow(reconstruction[0].squeeze(), cmap="gray" if nb_channels == 1 else None)
    axes[1].set_title("Reconstruction")
    axes[1].axis('off')

    axes[2].imshow(overlay)
    axes[2].set_title("Grad-CAM")
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(output_path / f"gradcam_{category}.png")
    plt.close(fig_plot)

    print(f"Grad-CAM sauvegardée dans {output_path / f'gradcam_{category}.png'}")

print("Terminé.")