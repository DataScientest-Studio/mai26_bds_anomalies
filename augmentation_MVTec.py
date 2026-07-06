from pathlib import Path
from dotenv import load_dotenv
from os import getenv
import sys
import cv2
import numpy as np

load_dotenv()

root_path = Path(getenv("PATH_DATASET"))

"""Arguments :
- (obligatoire) le nombre minimal d'images à avoir pour chaque type d'anomalie, 
- (optionnel) le nom du sous-dossier du dataset MVTec à traiter (tous si pas défini), 
- (optionnel) --train : si présent, le script va générer des images supplémentaires 
pour le sous-dossier train, sinon uniquement pour les anomalies du sous-dossier test.

Le script va parcourir les sous-dossiers du dataset MVTec et pour chaque type d'anomalie,
il va vérifier le nombre d'images présentes. Si ce nombre est inférieur au nombre minimal
d'images, le script va générer des images supplémentaires en appliquant des transformations aléatoires
sur les images existantes. Les images générées seront sauvegardées dans un sous-dossier
"augmented" du sous-dossier de l'anomalie. Si le sous-dossier "augmented" existe 
déjà, il sera vidé avant de sauvegarder les nouvelles images.
Les mêmes transformations seront appliquées aux masques correspondants, qui seront sauvegardés 
dans un sous-dossier "augmented" du sous-dossier "ground_truth".
Le script ne modifie pas les images originales.

Si '--train' est présent, le script procèdera de la même façon mais uniquement en parcourant le 
sous-dossier train et la qualité "good". (Il n'y a pas de masques pour les images du sous-dossier 
train, donc les transformations ne seront appliquées qu'aux images.)

Exemples : 
python augmentation_MVTec.py 100 bottle
python augmentation_MVTec.py 500 cable --train"""

# Certaines catégories du dataset MVTec ne peuvent pas recevoir certaines 
# transformations (ex: cable ne peut pas être retourné verticalement)
category_exceptions = {
    'cable': ['flip', 'rotate'],
    'capsule': ['flip', 'rotate'],
    'metal_nut': ['flip'],
    'pill': ['flip', 'rotate'],
    'transistor': ['flip', 'rotate'],
    'wood': ['flip', 'rotate'],
    'zipper': ['flip', 'rotate'],
}

# Vérification des arguments
if len(sys.argv) > 1:
    min_images = int(sys.argv[1])
else:
    # raise error if no argument is provided
    raise ValueError("Argument obligatoire : nombre minimal d'images pour chaque type d'anomalie.")
    
if len(sys.argv) > 2:
    # Je vérifie si c'est une catégorie ou --train
    if sys.argv[2] == "--train":
        # Je vérifie si le troisième argument est présent et si c'est une catégorie
        if len(sys.argv) > 3:
            category = sys.argv[3]
            train_mode = True
        else:
            category = None
            train_mode = True
    else:
        category = sys.argv[2]
        # Vérification du troisième argument
        if len(sys.argv) > 3 and sys.argv[3] == "--train":
            train_mode = True
        else:
            train_mode = False
else:
    category = None
    train_mode = False


def randomize_image(category, image_shape, zoom_max=1.08):
    """ Décide des transformations aléatoires à appliquer à l'image. 
    Renvoie un dictionnaire avec les transformations à appliquer """

    zoom = np.random.uniform(1.01, zoom_max)
    crop_w = int(image_shape[1] / zoom)
    crop_h = int(image_shape[0] / zoom)
    x_start = np.random.randint(0, image_shape[1] - crop_w + 1)
    y_start = np.random.randint(0, image_shape[0] - crop_h + 1)
    x_end = x_start + crop_w
    y_end = y_start + crop_h

    has_apply=False
    while has_apply==False:
        result = {
            "flip": {
                "apply": np.random.randint(3)==0, # 1 chance sur 3 d'appliquer un flip
                "type": np.random.randint(0,3),
            },
            "rotate": {
                "apply": np.random.randint(2)==0, # 1 chance sur 2 d'ajouter une rotation
                "angle": np.random.choice([180,90,-90],1)[0],
            },
            "noise": {
                "apply": np.random.randint(3)==0, # 1 chance sur 3 d'ajouter du bruit
                "noise": np.random.normal(0, 2, image_shape).astype(np.uint8),
            },
            "brightness": {
                "apply": np.random.randint(2)==0, # 1 chance sur 2 de changer la luminosité
                "value": np.random.randint(-25, 25),
            },
            "zoom": {
                "apply": np.random.randint(2)==0, # 1 chance sur 2 d'ajouter un zoom
                "value": zoom, 
                "x_start": x_start, 
                "y_start": y_start, 
                "x_end": x_end,
                "y_end": y_end,
            },
        }

        # on désactive les transformations interdites pour certaines catégories
        if category in category_exceptions:
            for transformation in category_exceptions[category]:
                result[transformation]['apply']=False
    
        # Vérifier qu'on a au moins 1 transformation qui va s'appliquer
        has_apply = any(v["apply"] for v in result.values())

    return result

def flip(image, flip_type):
    """ Retourne l'image retournée selon le code spécifié :
    - 0 : retournement vertical
    - 1 : retournement horizontal
    - 2 : retournement les deux
    Renvoie l'image modifiée"""
    # Retournement aléatoire
    if flip_type <= 1: # si 0 ou 1, flip vertical ou horizontal
        augmented_image = cv2.flip(image, flip_type)
    elif flip_type==2: # si 2, flip horizontal et vertical
        augmented_image = cv2.flip(image, 0)
        augmented_image = cv2.flip(image, 1)
    return augmented_image

def rotate(image, angle):
    """ Applique une rotation à l'image.
    Renvoie l'image modifiée"""
    
    M = cv2.getRotationMatrix2D((image.shape[1] / 2, image.shape[0] / 2), angle, 1)
    augmented_image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    return augmented_image

def gaussian_noise(image, noise):
    """ Ajoute du bruit à l'image.
    Renvoie l'image modifiée"""
    
    augmented_image = cv2.addWeighted(image, 1, noise, 0.05, 0)
    return augmented_image

def brightness(image, value):
    """ Change la luminosité de l'image.
    Renvoie l'image modifiée"""
    
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = cv2.add(hsv[:, :, 2], value)
    augmented_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return augmented_image

def zoom(image, x_start, x_end, y_start, y_end):
    """ Applique le zoom demandé sur l'image et resize l'image zoomée à la taille originale.
    Renvoie l'image modifiée"""
    h, w = image.shape[:2]

    augmented_image = image[y_start:y_end, x_start:x_end]
    augmented_image = cv2.resize(augmented_image, (w, h))
    return augmented_image

def transform_image(image, rnd_params, is_mask=False):
    """ Applique les transformations aléatoires à l'image selon les paramètres spécifiés.
    Renvoie l'image modifiée"""
    if rnd_params["flip"]["apply"]:
        image=flip(image, rnd_params["flip"]["type"])

    if rnd_params["rotate"]["apply"]:
        image=rotate(image, rnd_params["rotate"]["angle"])

    if rnd_params["noise"]["apply"] and not is_mask:
        image=gaussian_noise(image, rnd_params["noise"]["noise"])

    if rnd_params["brightness"]["apply"] and not is_mask:
        image=brightness(image, rnd_params["brightness"]["value"])
    
    if rnd_params["zoom"]["apply"]:
        image=zoom(image, rnd_params["zoom"]["x_start"], rnd_params["zoom"]["x_end"], 
                    rnd_params["zoom"]["y_start"], rnd_params["zoom"]["y_end"])
    return image

def augment_image(category, image_path, mask_path, output_directory, mask_output_directory, num_augmentations=5, copy_original=True):
    """ Applique des transformations aléatoires à une image et sauvegarde les images augmentées dans un dossier de sortie.
    - image_path : chemin vers l'image à augmenter
    - mask_path : chemin vers le masque de l'image à augmenter
    - output_directory : chemin vers le dossier où sauvegarder les images augmentées
    - mask_output_directory : chemin vers le dossier où sauvegarder les masques augmentés
    - num_augmentations : nombre d'images augmentées à générer (par défaut 5)"""
    image_init = cv2.imread(str(image_path))
    if mask_path is not None:
        mask_init = cv2.imread(str(mask_path))

    # Récupération du nom de l'image avec l'extension
    image_name = Path(image_path).stem
    image_extension = Path(image_path).suffix
    if mask_path is not None:
        mask_name = Path(mask_path).stem
        mask_extension = Path(mask_path).suffix

    # Copie de l'image si copy_original=True
    if copy_original == True:
        output_file_path = Path.joinpath(output_directory, f"{image_name}{image_extension}")
        cv2.imwrite(output_file_path, image_init)

        if mask_path is not None:
            output_mask_file_path = Path.joinpath(mask_output_directory, f"{mask_name}{mask_extension}")
            cv2.imwrite(output_mask_file_path, mask_init)
    
    for i in range(num_augmentations):
        image=image_init.copy()
        rnd_params = randomize_image(category, image.shape)

        # Image
        image=transform_image(image, rnd_params)
        # Enregistrer l'image augmentée
        output_file_path = Path.joinpath(output_directory, f"{image_name}_{i:02d}{image_extension}")
        cv2.imwrite(output_file_path, image)

        # Masque
        if mask_path is not None:
            mask=mask_init.copy()
            mask=transform_image(mask, rnd_params, is_mask=True)
            # Enregistrer le masque augmenté
            output_mask_file_path = Path.joinpath(mask_output_directory, f"{mask_name}_{i:02d}{mask_extension}")
            cv2.imwrite(output_mask_file_path, mask)

def empty_directory(directory):
    """ Vide le contenu d'un dossier et le crée s'il n'existe pas.
     - directory : chemin vers le dossier à vider"""
    Path(directory).mkdir(parents=True, exist_ok=True)
    for file in directory.iterdir():
        if file.is_file():
            file.chmod(0o666)
            file.unlink()

if category is None:
    # On crée un tableau categories avec tous les noms de sous-dossiers du dataset MVTec
    categories = [cat.name for cat in root_path.iterdir() if cat.is_dir()]
else:
    # On vérifie que la catégorie spécifiée existe dans le dataset MVTec
    assert root_path.joinpath(category).is_dir(), Exception("Categorie introuvable: "+category)
    categories = [category]

for i, category in enumerate(categories):
    print(f"Traitement de la catégorie : {category} ({i+1}/{len(categories)})")
    
    if train_mode:
        # Répertoire des images de train
        dir_images = Path.joinpath(root_path, category, "train")
        # Répertoire des masques de train (inexistant)
        dir_masks = None
        defects=['good']
    else:
        # Répertoire des images de test
        dir_images = Path.joinpath(root_path, category, "test")
        dir_masks = Path.joinpath(root_path, category, "ground_truth")
        
        defects = [defect.name for defect in dir_images.iterdir() if defect.is_dir() and defect.name != "good"]

    for j, defect in enumerate(defects):
        # Compter les images
        dir_defect = Path.joinpath(dir_images, defect)
        if dir_masks is not None:
            dir_defect_mask = Path.joinpath(dir_masks, defect)
        images = [(image.stem, image.suffix) for image in dir_defect.iterdir() if image.is_file()]
        
        nb_images_to_create = max(0, min_images - len(images))
        nb_augmented_images_per_image = max(0, nb_images_to_create // len(images)) + (1 if nb_images_to_create % len(images) > 0 else 0)
        print(f"\nTraitement de l'anomalie : {defect} ({j+1}/{len(defects)}) (création de {nb_augmented_images_per_image * len(images)} images supplémentaires, {nb_augmented_images_per_image} par image existante)")
        
        # Vide les dossiers 'augmented'
        augmented_dir = Path.joinpath(dir_defect, 'augmented')
        empty_directory(augmented_dir)
        if dir_masks is not None:
            augmented_mask_dir = Path.joinpath(dir_defect_mask, 'augmented')
            empty_directory(augmented_mask_dir)
        else:
            augmented_mask_dir = None
        
        for image in images:
            # Progress bar
            progress = (images.index(image) + 1) / len(images) * 100
            print(f"\rProgression : [{'#' * int(progress // 2)}{' ' * (50 - int(progress // 2))}] {progress:.2f}%", end='')
            
            mask_path = Path.joinpath(dir_defect_mask, f"{image[0]}_mask{image[1]}") if dir_masks is not None else None

            augment_image(category=category, 
                        image_path=Path.joinpath(dir_defect, f"{image[0]}{image[1]}"),
                        mask_path=mask_path,
                        output_directory=augmented_dir,
                        mask_output_directory=augmented_mask_dir,
                        num_augmentations=nb_augmented_images_per_image, 
                        copy_original=True)