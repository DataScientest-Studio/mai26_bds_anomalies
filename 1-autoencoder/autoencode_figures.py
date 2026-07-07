import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc

# Visualisation des images reconstruites
def compare_orig_encoded(orig_images, encoded_images, output_path, output_filename="images_reconstruites_train_good.png"):
    """ Affiche les 6 premières images originales, leur version auto-encodées et leurs différences.
    """
    nb_col = 6

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
            plt.imshow( image_originale, cmap="gray" , vmin=0, vmax=1)
        else:
            plt.imshow( image_originale, vmin=0, vmax=1)
        plt.axis('off')
        plt.title("Original")
        
        plt.subplot(3,nb_col, i+1+nb_col)
        if (image_autoencodee.ndim == 2) or (image_originale.shape[2] == 1):
            plt.imshow( image_autoencodee, cmap="gray" , vmin=0, vmax=1)
        else:
            plt.imshow( image_autoencodee, vmin=0, vmax=1)
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
def histogramme_erreurs(orig_images_flat, encoded_images_flat, y_true, threshold, output_path, output_filename="histogramme_erreurs.png", category=""):
    mse = ((orig_images_flat - encoded_images_flat)**2).mean(axis=1)

    plt.figure(figsize=(10,6))
    plt.hist(mse[y_true==0], bins=30, color='green', alpha=0.7, label='Train (bonnes)')
    plt.xlabel('Erreur')
    plt.ylabel('Fréquence')
    plt.title(f'Histogramme des erreurs - {category}')

    # Définition d'une limite pour détecter les anomalies (par exemple, le 95ème percentile des erreurs sur les images d'entraînement)
    plt.axvline(threshold, color='red', linestyle='dashed', linewidth=2, label=f'Seuil (95ème percentile) = {threshold:.5f}')

    # Histogramme des erreurs sur les images de test (anomalies)
    plt.hist(mse[y_true==1], bins=30, color='orange', alpha=0.7, label='Test (anomalies)')
    plt.legend()
    plt.savefig(output_path / output_filename)

def draw_confusion_matrix(y_true, y_pred, output_path, output_filename="matrice_confusion.png", category=""):
    plt.figure(figsize=(8,6))
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    sns.heatmap(cm, annot=True, fmt='.3f', cmap='Blues', xticklabels=['Good', 'Anomalies'], yticklabels=['Good', 'Anomalies'])
    plt.xlabel('Prédiction')
    plt.ylabel('Réel')
    plt.title(f'Matrice de confusion - {category}')
    plt.savefig(output_path / output_filename)

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

"""

# ROC curve
from sklearn.metrics import roc_curve, auc

fpr, tpr, thresholds = roc_curve(np.concatenate([np.zeros(len(mse_train_good)), np.ones(len(mse_anomaly))]), np.concatenate([mse_train_good, mse_anomaly]))
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
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.savefig(output_path / f"roc_curve.png")

# Visualisation des images reconstruites pour les anomalies
# Visualisation des images reconstruites
nb_col = 6
test_pred = autoencoder.predict(images_test_anomaly[:nb_col])

plt.figure(figsize=(14,8))
for i in range(nb_col):
    
    image_originale = images_test_anomaly[i].reshape(resized_dimension[0], resized_dimension[1], nb_channels_test_anomaly)
    image_autoencodee = test_pred[i].reshape(resized_dimension[0], resized_dimension[1], nb_channels_test_anomaly)
    image_erreur = np.abs(image_originale - image_autoencodee)
    
    plt.subplot(3,nb_col, i+1)
    plt.imshow( image_originale, cmap="gray" , vmin=0, vmax=1)
    plt.axis('off')
    plt.title("Original")
    
    plt.subplot(3,nb_col, i+1+nb_col)
    plt.imshow( image_autoencodee , cmap="gray" , vmin=0, vmax=1)
    plt.axis('off')
    plt.title("Auto-encodé")
    
    plt.subplot(3,nb_col, i+1+nb_col*2)
    plt.imshow( image_erreur , cmap="hot" , vmin=0, vmax=1)
    plt.axis('off')
    mae = np.mean(np.abs(image_originale - image_autoencodee))
    mse = np.mean((image_originale - image_autoencodee) ** 2)

    plt.title(f"Erreur\nMAE={mae:.5f}\nMSE={mse:.5f}")
    
plt.savefig(output_path / "images_reconstruites_test_anomalies.png")"""