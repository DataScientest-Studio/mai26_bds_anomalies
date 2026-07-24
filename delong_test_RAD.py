import numpy as np
import pandas as pd
import joblib
import os
import sys 
from pathlib import Path
from itertools import combinations
from scipy import stats
sys.path.append(str(Path(__file__).parent / "VAE_transfer"))
from vae_transfer_model import create_model, save_history_plot  # nécessaire pour enregistrer Sampling/KLLossLayer avant joblib.load
from vae_transfer_load_images_RAD import load_liste_images

from dotenv import load_dotenv

load_dotenv('sample.env')
image_path = Path(os.getenv("PATH_DATASET_RAD"))

resized_dimension = (64,64)
categories = ['bolt', 'ribbon', 'sponge', 'tape']

# dictionnaire : nom du modèle -> dossier output correspondant
# adapte les chemins selon l'endroit où sont sauvegardés tes .joblib pour chaque architecture
model_variants = {
    "VAE_simple": Path(__file__).parent.joinpath("output", "VAE_1"),
    "VAE_transfer": Path(__file__).parent.joinpath("output", "VAE_transfer"),
    # ajoute d'autres variantes ici si besoin
}

results_csv_path = Path(__file__).parent.parent / "delong_results.csv"


def compute_auc_variance_delong(y_true, scores):
    """
    Calcule l'AUC et sa variance selon DeLong, en comparant directement 
    chaque paire (positif, négatif) - version simple, non optimisée mais 
    largement suffisante pour de petits datasets.
    """
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)

    pos_scores = scores[y_true == 1]
    neg_scores = scores[y_true == 0]
    m = len(pos_scores)
    n = len(neg_scores)

    # matrice de comparaison : psi[i,j] = 1 si pos[i] > neg[j], 0.5 si égal, 0 sinon
    psi = np.zeros((m, n))
    for i in range(m):
        psi[i, :] = (pos_scores[i] > neg_scores).astype(float) + 0.5 * (pos_scores[i] == neg_scores)

    auc = psi.mean()

    # variance selon la méthode de DeLong (structure Mann-Whitney U)
    v10 = psi.mean(axis=1)  # moyenne sur les négatifs, pour chaque positif
    v01 = psi.mean(axis=0)  # moyenne sur les positifs, pour chaque négatif

    var_auc = v10.var(ddof=1) / m + v01.var(ddof=1) / n

    return auc, var_auc


def delong_roc_test(y_true, score_A, score_B):
    """
    Compare deux jeux de scores (score_A, score_B) sur les mêmes y_true. 
    Retourne (auc_A, auc_B, p_value).
    """
    auc_A, var_A = compute_auc_variance_delong(y_true, score_A)
    auc_B, var_B = compute_auc_variance_delong(y_true, score_B)

    # covariance entre les deux AUC (car mesurées sur les mêmes exemples)
    y_true = np.asarray(y_true)
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]

    # approximation simple : corrélation entre les scores sur positifs et négatifs
    cov_pos = np.cov(score_A[pos_idx], score_B[pos_idx])[0, 1] if len(pos_idx) > 1 else 0
    cov_neg = np.cov(score_A[neg_idx], score_B[neg_idx])[0, 1] if len(neg_idx) > 1 else 0

    # NB: ceci est une approximation ; la covariance exacte de DeLong entre 
    # deux AUC nécessite normalement de repasser par les mêmes matrices psi -
    # pour un calcul rigoureux, garder la version "fast" est préférable
    var_diff = var_A + var_B  # approximation simple si on ignore la covariance

    z = (auc_A - auc_B) / np.sqrt(var_diff)
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))

    return auc_A, auc_B, p_value


results = []

for category in categories:

    print(f"\n=== Catégorie : {category} ===")

    # Chargement UNIQUE des images de test, pour garantir que tous les modèles 
    # sont évalués sur exactement les mêmes images, dans le même ordre
    images_test_good_flat, nb_channels_good = load_liste_images(
        image_path, resized_dimension, category=category, type='test', quality="good", include_augmented=False
    )
    images_test_anomaly_flat, nb_channels_anomaly = load_liste_images(
        image_path, resized_dimension, category=category, type='test', quality="anomaly", include_augmented=False
    )

    nb_channels = nb_channels_good  # suppose good et anomaly ont le même nb de canaux
    images_test_good = images_test_good_flat.reshape(-1, resized_dimension[0], resized_dimension[1], nb_channels)
    images_test_anomaly = images_test_anomaly_flat.reshape(-1, resized_dimension[0], resized_dimension[1], nb_channels)

    y_true = np.concatenate([
        np.zeros(len(images_test_good_flat)), 
        np.ones(len(images_test_anomaly_flat))
    ])

    # Calcul des scores (MSE) pour chaque variante de modèle disponible
    model_scores = {}

    for model_name, model_output_path in model_variants.items():
        model_file = model_output_path / f"autoencoder_{category}.joblib"

        if not model_file.is_file():
            print(f"  Modèle '{model_name}' non trouvé pour '{category}' ({model_file}), skip.")
            continue

        autoencoder = joblib.load(model_file)

        pred_test_good = autoencoder.predict(images_test_good)
        pred_test_good_flat = pred_test_good.reshape(-1, resized_dimension[0]*resized_dimension[1]*nb_channels)
        mse_test_good = ((images_test_good_flat - pred_test_good_flat)**2).mean(axis=1)

        pred_test_anomaly = autoencoder.predict(images_test_anomaly)
        pred_test_anomaly_flat = pred_test_anomaly.reshape(-1, resized_dimension[0]*resized_dimension[1]*nb_channels)
        mse_test_anomaly = ((images_test_anomaly_flat - pred_test_anomaly_flat)**2).mean(axis=1)

        scores = np.concatenate([mse_test_good, mse_test_anomaly])
        model_scores[model_name] = scores

        print(f"  Scores calculés pour '{model_name}'")

    # Comparaison de chaque paire de modèles disponibles pour cette catégorie
    available_models = list(model_scores.keys())

    if len(available_models) < 2:
        print(f"  Pas assez de modèles disponibles pour comparer sur '{category}' (il en faut au moins 2).")
        continue

    for model_A, model_B in combinations(available_models, 2):
        auc_A, auc_B, p_value = delong_roc_test(
            y_true, model_scores[model_A], model_scores[model_B]
        )

        significatif = p_value < 0.05

        print(f"  {model_A} (AUC={auc_A:.4f}) vs {model_B} (AUC={auc_B:.4f}) "
              f"-> p-value={p_value:.4f} {'(significatif)' if significatif else '(non significatif)'}")

        results.append({
            "category": category,
            "model_A": model_A,
            "model_B": model_B,
            "auc_A": auc_A,
            "auc_B": auc_B,
            "p_value": p_value,
            "significatif_0.05": significatif,
        })

# Sauvegarde de tous les résultats dans un CSV
results_df = pd.DataFrame(results)
results_df.to_csv(results_csv_path, index=False)
print(f"\nRésultats complets sauvegardés dans {results_csv_path}")
print(results_df)