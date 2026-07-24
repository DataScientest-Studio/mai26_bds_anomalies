import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve, auc
import tensorflow as tf
from tensorflow.keras.models import Model

from functools import wraps
import cv2

DEBUG=False

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
def compare_orig_encoded(image_dataset, model, output_path, output_filename="images_reconstruites_train_good.png", 
                         grad_layer_name=None, nb_min_images=5, all_classes=False, class_names={0:"good"}):
    """ Affiche les num_images premières images originales, leur grad-cam, leur version auto-encodées et leurs différences (MSE).
    Si only_label est défini, on n'affichera que des images qui ont ce label.
    """

    if all_classes is None or all_classes==False:
        for images, labels in image_dataset.take(1):
            orig_images = images[:nb_min_images]
            orig_labels = labels[:nb_min_images].numpy()

    else:
        orig_images = []
        orig_labels = []
        nb_classes = len(class_names)
        for images, labels in image_dataset:
            for image, label in zip(images, labels):
                if label not in orig_labels:
                    orig_images.append(image)
                    orig_labels.append(label)
                    if len(orig_images) >= nb_classes:
                        break
            if len(orig_images) >= nb_classes:
                break
        orig_images = tf.stack(orig_images, axis=0)
        orig_labels = np.array(orig_labels)

    if grad_layer_name is None or grad_layer_name == "":
        encoded_images = model.predict(orig_images)
        grad_layer_name = None
    else:
        heatmaps, encoded_images = make_gradcam_heatmap(orig_images, model, last_conv_layer_name=grad_layer_name)

    nb_images = len(orig_images)
    plt.figure(figsize=(14,3*nb_images))
    for i in range(nb_images):
        
        image_originale = orig_images[i]
        image_autoencodee = encoded_images[i]

        if orig_labels.ndim == 1:
            image_label = class_names[orig_labels[i]]
        else:
            image_label = 'good'

        if image_originale.ndim == 3 and image_originale.shape[2] > 1:
            image_erreur_mse = np.mean((image_originale - image_autoencodee)**2, axis=2)
            image_erreur_mae = np.abs(image_originale - image_autoencodee).mean(axis=2)
        else:
            image_erreur_mse = (image_originale - image_autoencodee)**2
            image_erreur_mae = np.abs(image_originale - image_autoencodee)

        n_col = 5
        if grad_layer_name is None:
            n_col = 4
        subplot_index = (i*n_col)+1
        
        plt.subplot(nb_images,n_col, subplot_index)
        if (image_originale.ndim == 2) or (image_originale.shape[2] == 1):
            plt.imshow( image_originale, cmap="gray")
        else:
            plt.imshow( image_originale)
        plt.axis('off')
        plt.title(image_label)
        subplot_index+=1

        if grad_layer_name is not None:
            plt.subplot(nb_images,n_col, subplot_index)
            overlay = overlay_heatmap(image_originale, heatmaps[i])
            plt.imshow( overlay)
            plt.axis('off')
            plt.title("Grad-cam")
            subplot_index+=1
        
        plt.subplot(nb_images,n_col, subplot_index)
        if (image_autoencodee.ndim == 2) or (image_originale.shape[2] == 1):
            plt.imshow( image_autoencodee, cmap="gray")
        else:
            plt.imshow( image_autoencodee)
        plt.axis('off')
        plt.title("Auto-encodé")
        subplot_index+=1
        
        plt.subplot(nb_images,n_col, subplot_index)
        plt.imshow( image_erreur_mae , cmap="hot" , vmin=0, vmax=1)
        plt.axis('off')
        mae = np.mean(np.abs(image_originale - image_autoencodee))
        mse = np.mean((image_originale - image_autoencodee) ** 2)

        plt.title(f"MAE={mae:.5f}")
        subplot_index+=1
        
        plt.subplot(nb_images,n_col, subplot_index)
        plt.imshow( image_erreur_mse , cmap="hot" , vmin=0, vmax=1)
        plt.axis('off')
        mae = np.mean(np.abs(image_originale - image_autoencodee))
        mse = np.mean((image_originale - image_autoencodee) ** 2)

        plt.title(f"MSE={mse:.5f}")
        subplot_index+=1
        
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
    plt.axvline(threshold, color='red', linestyle='dashed', linewidth=2, label=f'Seuil = {threshold:.5f}')

    
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

def tensor_stats(name, tensor):
    print(
        f"{name:25s}",
        f"shape={tensor.shape}",
        f"min={tf.reduce_min(tensor).numpy():.3e}",
        f"max={tf.reduce_max(tensor).numpy():.3e}",
        f"mean={tf.reduce_mean(tensor).numpy():.3e}",
        f"abs_mean={tf.reduce_mean(tf.abs(tensor)).numpy():.3e}",
        f"zeros={tf.reduce_mean(tf.cast(tensor == 0, tf.float32)).numpy():.2%}",
    )

def make_gradcam_heatmap(img_array, autoencoder, last_conv_layer_name):
    #encoder = autoencoder.get_layer(encoder_model_name)
    last_conv_layer = autoencoder.get_layer(last_conv_layer_name)

    grad_model = tf.keras.Model(
        autoencoder.inputs, [last_conv_layer.output, autoencoder.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, reconstructions = grad_model(img_array, training=False)

        losses = tf.reduce_mean(tf.square(img_array - reconstructions), axis=(1,2,3))

    grads = tape.gradient(losses, conv_outputs)
    if grads is None:
        raise ValueError(f"Les gradients ne peuvent pas être calculés pour la couche {last_conv_layer_name}'.")

    pooled_grads = tf.reduce_mean(grads, axis=(1, 2))
    
    #heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    # pour une image devient ça pour un batch :
    weights = pooled_grads[:, tf.newaxis, tf.newaxis, :]
    weighted_activations = conv_outputs * weights
    raw_heatmaps = tf.reduce_sum(
        weighted_activations,
        axis=-1
    )

    heatmaps = tf.nn.relu(raw_heatmaps)

    #heatmap /= (tf.math.reduce_max(heatmap) + 1e-8)
    # pour une image devient ça pour un batch :
    max_values = tf.reduce_max(
        heatmaps,
        axis=(1, 2),
        keepdims=True
    )
    heatmaps = tf.math.divide_no_nan(
        heatmaps,
        max_values
    )
    if DEBUG:
        for i in range(len(conv_outputs)):
            print(f"--- DEBUG image {i} ---")
            tensor_stats("conv_outputs", conv_outputs[i])
            tensor_stats("grads", grads[i])
            tensor_stats("pooled_grads", pooled_grads[i])
            tensor_stats("raw_heatmaps", raw_heatmaps[i])
            tensor_stats("heatmaps après ReLU", heatmaps[i])

    return heatmaps.numpy(), reconstructions.numpy()

def overlay_heatmap(image, heatmap, alpha=0.4):
    # Conversion TensorFlow vers NumPy si nécessaire
    if hasattr(image, "numpy"):
        image = image.numpy()
    image = np.asarray(image, dtype=np.float32)

    heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    #heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_colored = plt.cm.jet(heatmap_resized)[...,:3].astype(np.float32)
    #heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    if image.shape[-1] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    overlay = cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)

    return overlay