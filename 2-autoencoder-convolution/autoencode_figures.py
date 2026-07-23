import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve, auc
import tensorflow as tf

from functools import wraps
import cv2

def logging_function(function):
    @wraps(function)
    def printing(*args, **kwargs):
        # print(f"{function.__name__} starting")
        result = function(*args, **kwargs)
        print(f"{function.__name__} finished")
        return result
    return printing

# Visualisation des images reconstruites
@logging_function
def compare_orig_encoded(image_dataset, model, output_path, output_filename="images_reconstruites_train_good.png", only_label=None):
    """ Affiche les 6 premières images originales, leur version auto-encodées et leurs différences.
    """
    nb_col = 6

    if only_label is None:
        for images, labels in image_dataset.take(1):
            orig_images = images[:nb_col]

    else:
        orig_images = []
        for images, labels in image_dataset:
            for image, label in zip(images, labels):
                if label == only_label:
                    orig_images.append(image)
                    if len(orig_images) >= nb_col:
                        break
            if len(orig_images) >= nb_col:
                break
        orig_images = tf.stack(orig_images, axis=0)

    encoded_images = model.predict(orig_images)

    plt.figure(figsize=(14,8))
    for i in range(nb_col):
        
        image_originale = orig_images[i]
        image_autoencodee = encoded_images[i]

        if image_originale.ndim == 3 and image_originale.shape[2] > 1:
            image_erreur = np.abs(image_originale - image_autoencodee).mean(axis=2)
        else:
            image_erreur = np.abs(image_originale - image_autoencodee)
        
        plt.subplot(3,nb_col, i+1)
        if (image_originale.ndim == 2) or (image_originale.shape[2] == 1):
            plt.imshow( image_originale, cmap="gray")
        else:
            plt.imshow( image_originale)
        plt.axis('off')
        plt.title("Original")
        
        plt.subplot(3,nb_col, i+1+nb_col)
        if (image_autoencodee.ndim == 2) or (image_originale.shape[2] == 1):
            plt.imshow( image_autoencodee, cmap="gray")
        else:
            plt.imshow( image_autoencodee)
        plt.axis('off')
        plt.title("Auto-encodé")
        
        plt.subplot(3,nb_col, i+1+nb_col*2)
        plt.imshow( image_erreur , cmap="hot" , vmin=0, vmax=1)
        plt.axis('off')
        mae = np.mean(np.abs(image_originale - image_autoencodee))
        mse = np.mean((image_originale - image_autoencodee) ** 2)

        plt.title(f"Erreur\nMAE={mae:.5f}\nMSE={mse:.5f}")
        
    plt.savefig(output_path / output_filename)


# Histogramme des erreurs sur les images d'entraînement (bonnes)
@logging_function
def histogramme_erreurs(train_mses, 
                        test_mses, test_labels, 
                        threshold, 
                        output_path, output_filename="histogramme_erreurs.png", category=""):

    if not isinstance(train_mses, np.ndarray):
        train_mses = np.array(train_mses)
    if not isinstance(test_mses, np.ndarray):
        test_mses = np.array(test_mses)
    if not isinstance(test_labels, np.ndarray):
        test_labels = np.array(test_labels)

    plt.figure(figsize=(10,6))
    # Anomalies (test)
    plt.hist(test_mses[test_labels==1], bins=30, color='orange', alpha=0.7, label='Test (anomalies)')
    # Good (train)
    plt.hist(train_mses, bins=30, color='lightgreen', alpha=0.7, label='Train (good)')
    # Good (test)
    plt.hist(test_mses[test_labels==0], bins=30, color='green', alpha=0.7, label='Test (good)')

    plt.xlim(
        min(*train_mses, *test_mses), max(*train_mses, *test_mses)
    )
    plt.xlabel('Erreur')
    plt.ylabel('Fréquence')
    plt.title(f'Histogramme des erreurs - {category}')

    # Définition d'une limite pour détecter les anomalies (par exemple, le 95ème percentile des erreurs sur les images d'entraînement)
    plt.axvline(threshold, color='red', linestyle='dashed', linewidth=2, label=f'Seuil (95ème percentile) = {threshold:.5f}')

    
    plt.legend()
    plt.savefig(output_path / output_filename)

@logging_function
def draw_confusion_matrix(y_true, y_pred, output_path, output_filename="matrice_confusion.png", category=""):
    plt.figure(figsize=(8,6))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Good', 'Anomalies'], yticklabels=['Good', 'Anomalies'])
    plt.xlabel('Prédiction')
    plt.ylabel('Réel')
    plt.title(f'Matrice de confusion - {category}')
    plt.savefig(output_path / output_filename)

    return cm

@logging_function
def save_classification_report(y_true, y_pred, output_path, output_filename="classification_report.txt", comment="", append=False):
    with open(output_path / output_filename, "a" if append else "w") as f:
        f.write(f"--- Classification report - {comment} ---\n")
        f.write(classification_report(y_true, y_pred, zero_division=0,))
        f.write("\n\n")

@logging_function
def draw_roc_curve(mses, labels, output_path, output_filename="roc_curve.png", category=""):

    fpr, tpr, thresholds = roc_curve(labels, mses)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8,6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.yticks(np.arange(0, 1.06, 0.05))
    plt.grid()
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f"ROC Curve - {category}")
    plt.legend(loc="lower right")
    plt.savefig(output_path / output_filename)

    return roc_auc


def make_gradcam_heatmap(img_array, autoencoder, last_conv_layer_name, encoder_model_name="encodeur"):
    #encoder = autoencoder.get_layer(encoder_model_name)
    last_conv_layer = autoencoder.get_layer(last_conv_layer_name)

    grad_model = tf.keras.Model(
        autoencoder.inputs, [last_conv_layer.output, autoencoder.output]
    )

    with tf.GradientTape() as tape:
        conv_output, reconstruction = grad_model(img_array)

        loss = tf.reduce_mean(tf.square(img_array - reconstruction))

    grads = tape.gradient(loss, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0)
    heatmap /= (tf.math.reduce_max(heatmap) + 1e-8)

    return heatmap.numpy(), reconstruction.numpy()

def overlay_heatmap(image, heatmap, alpha=1):
    heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    image_uint8 = np.uint8(255 * image) if np.max(image) <= 1 else image.astype(np.uint8)
    if image_uint8.shape[-1] == 1:
        image_uint8 = cv2.cvtColor(image_uint8, cv2.COLOR_GRAY2RGB)

    overlay = cv2.addWeighted(image_uint8, 1 - alpha, heatmap_colored, alpha, 0)

    return overlay