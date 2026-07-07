# Script pour générer un schéma qui montre une image sans défaut pour chaque catégorie
# et un schéma par catégorie qui montre une image sans défaut + une image de chaque type d'anomalie

import pandas as pd
import cv2
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from pathlib import Path
import os

load_dotenv()
image_path_mvtec = Path(os.getenv("PATH_DATASET"))
image_path_rad = Path(os.getenv("PATH_DATASET_RAD"))

# output = "output/images_by_category" (créer le répertoire s'il n'existe pas)
os.makedirs("output/images_by_category", exist_ok=True)
output_path = Path("output/images_by_category")

image_list = pd.read_csv('image_list_clean.csv', dtype={'file': str})

def generate_complete_path(origin, image_category, image_type, image_quality, image_filename):
    if origin.lower() == "mvtec":
        return image_path_mvtec.joinpath(image_category, image_type, image_quality, str(image_filename) + ".png")
    else:
        if image_quality == 'good':
            return image_path_rad.joinpath('bolt', image_type, 'good', str(image_filename) + ".png")
        else:
            return image_path_rad.joinpath(image_quality, image_type, 'defect', str(image_filename) + ".png")


################################################################################
# Générer un schéma avec une image aléatoire sans défaut pour chaque catégorie #
################################################################################

# Compter les catégories uniques
categories = image_list["category"].unique()

nb_lignes_affichees = (len(categories)-1) // 6 + 1
plt.figure(figsize=(16, 3*nb_lignes_affichees))

for i, category in enumerate(categories):
    plt.subplot(nb_lignes_affichees, 6, i + 1)
    # Filtrer les images de la catégorie et sans défaut
    category_images = image_list[(image_list["category"] == category) & (image_list["quality"] == "good")]

    # Sélectionner une image aléatoire
    random_image = category_images.sample(n=1)

    # Afficher l'image
    img_path = generate_complete_path(random_image.iloc[0]["origin"], category, random_image.iloc[0]["type"], random_image.iloc[0]["quality"], random_image.iloc[0]["file"])
    try:
        img = cv2.imread(img_path)

        # Si l'image est en niveau de gris, la convertir en RGB pour l'affichage
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        plt.imshow(img)
    except Exception as e:
        print(f"Erreur lors de la lecture de l'image {img_path}: {e}")
        continue

    plt.title(category)
    plt.axis('off')

plt.savefig(output_path / "images_by_category.png")
print(f"images_by_categories.png généré ({len(categories)} catégories)")

############################################################################
# Générer un schéma par catégorie avec une image de chaque type d'anomalie #
############################################################################

for i, category in enumerate(categories):

    qualities = image_list.loc[image_list["category"] == category, "quality"].unique()
    
    nb_lignes_affichees = (len(qualities)-1) // 6 + 1
    plt.figure(figsize=(16, 3*nb_lignes_affichees))

    # Mettre la quality "good" au début de la liste
    qualities = ["good"] + [q for q in qualities if q != "good"]

    for i, quality in enumerate(qualities):
        plt.subplot(nb_lignes_affichees, 6, i + 1)
        # Filtrer les images de la catégorie et sans défaut
        category_images = image_list[(image_list["category"] == category) & (image_list["quality"] == quality)]

        # Sélectionner une image aléatoire
        random_image = category_images.sample(n=1)

        # Afficher l'image
        img_path = generate_complete_path(random_image.iloc[0]["origin"], category, random_image.iloc[0]["type"], random_image.iloc[0]["quality"], random_image.iloc[0]["file"])
        try:
            img = cv2.imread(img_path)
        
            # Si l'image est en niveau de gris, la convertir en RGB pour l'affichage
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            plt.imshow(img)
        except Exception as e:
            print(f"Erreur lors de la lecture de l'image {img_path}: {e}")
            continue

        plt.title(quality)
        plt.axis('off')

    plt.savefig(output_path / f"images_{category}.png")
    print(f"images_{category}.png généré ({len(qualities)} qualités)")