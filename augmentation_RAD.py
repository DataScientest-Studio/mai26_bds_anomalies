from pathlib import Path
from dotenv import load_dotenv
from os import getenv
import sys
import cv2
import numpy as np

load_dotenv('sample.env')

root_path = Path(getenv("PATH_DATASET_RAD"))

"""Le script reçoit en argument le nombre de variantes à générer par image
et peut recevoir le nom de la catégorie RAD à traiter (toutes si pas défini).

Le script augmente uniquement train/good.
test/good et test/defect ne sont pas modifiés.

"""

# Vérification des arguments
if len(sys.argv) > 1:
    nb_augmentations = int(sys.argv[1])
else:
    raise ValueError("Argument obligatoire : nombre de variantes à générer par image.")

if len(sys.argv) > 2:
    category = sys.argv[2]
else:
    category = None


#transformations

def randomize_image(image_shape, zoom_max=1.08):
    """Décide des transformations aléatoires à appliquer.
    Garantit qu'au moins une transformation est activée."""

    zoom = np.random.uniform(1.01, zoom_max)
    crop_w = int(image_shape[1] / zoom)
    crop_h = int(image_shape[0] / zoom)
    x_start = np.random.randint(0, image_shape[1] - crop_w + 1)
    y_start = np.random.randint(0, image_shape[0] - crop_h + 1)

    has_apply = False
    while not has_apply:
        result = {
            "flip": {
                "apply": np.random.randint(3) == 0,      
                "type": np.random.randint(0, 3),
            },
            "rotate": {
                "apply": np.random.randint(2) == 0,      
                "angle": np.random.choice([90, 180, -90], 1)[0],
            },
            "noise": {
                "apply": np.random.randint(3) == 0,      
                "noise": np.random.normal(0, 2, image_shape).astype(np.uint8),
            },
            "brightness": {
                "apply": np.random.randint(2) == 0,     
                "value": np.random.randint(-25, 25),
            },
            "zoom": {
                "apply": np.random.randint(2) == 0,      
                "x_start": x_start,
                "y_start": y_start,
                "x_end": x_start + crop_w,
                "y_end": y_start + crop_h,
            },
        }
        has_apply = any(v["apply"] for v in result.values())

    return result


def flip(image, flip_type):
    if flip_type <= 1:
        return cv2.flip(image, flip_type)
    else:
        img = cv2.flip(image, 0)
        return cv2.flip(img, 1)


def rotate(image, angle):
    M = cv2.getRotationMatrix2D((image.shape[1] / 2, image.shape[0] / 2), angle, 1)
    return cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))


def gaussian_noise(image, noise):
    return cv2.addWeighted(image, 1, noise, 0.05, 0)


def brightness(image, value):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = cv2.add(hsv[:, :, 2], value)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def zoom(image, x_start, x_end, y_start, y_end):
    h, w = image.shape[:2]
    cropped = image[y_start:y_end, x_start:x_end]
    return cv2.resize(cropped, (w, h))


def transform_image(image, rnd_params):
    """Applique les transformations selon les paramètres tirés aléatoirement."""
    if rnd_params["flip"]["apply"]:
        image = flip(image, rnd_params["flip"]["type"])

    if rnd_params["rotate"]["apply"]:
        image = rotate(image, rnd_params["rotate"]["angle"])

    if rnd_params["noise"]["apply"]:
        image = gaussian_noise(image, rnd_params["noise"]["noise"])

    if rnd_params["brightness"]["apply"]:
        image = brightness(image, rnd_params["brightness"]["value"])

    if rnd_params["zoom"]["apply"]:
        image = zoom(image,
                     rnd_params["zoom"]["x_start"], rnd_params["zoom"]["x_end"],
                     rnd_params["zoom"]["y_start"], rnd_params["zoom"]["y_end"])
    return image



#utilitaires 


def empty_directory(directory):
    """Vide un dossier et le crée s'il n'existe pas."""
    Path(directory).mkdir(parents=True, exist_ok=True)
    for file in Path(directory).iterdir():
        if file.is_file():
            file.chmod(0o666)
            file.unlink()



#pipeline principal

if category is None:
    categories = [cat.name for cat in root_path.iterdir() if cat.is_dir()]
else:
    assert root_path.joinpath(category).is_dir(), f"Catégorie introuvable : {category}"
    categories = [category]

EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}

for i, cat in enumerate(categories):
    print(f"\n{'='*60}")
    print(f"Catégorie : {cat} ({i+1}/{len(categories)})")
    print(f"{'='*60}")

    dir_train_good = root_path / cat / "train" / "good"

    if not dir_train_good.exists():
        print(f"   Dossier introuvable, catégorie ignorée : {dir_train_good}")
        continue

    images = [p for p in dir_train_good.iterdir() if p.suffix.lower() in EXTENSIONS and p.is_file()]

    if not images:
        print(f"   Aucune image trouvée dans {dir_train_good}")
        continue

    total_generated = len(images) * nb_augmentations
    print(f"  → {len(images)} images originales")
    print(f"  → {nb_augmentations} variantes par image")
    print(f"  → {len(images) + total_generated} images au total après augmentation")

    # Vide et recrée le dossier augmented
    out_dir = dir_train_good / "augmented"
    empty_directory(out_dir)

    for j, image_path in enumerate(images):
        # Barre de progression
        progress = (j + 1) / len(images) * 100
        print(f"\rProgression : [{'#' * int(progress // 2)}{' ' * (50 - int(progress // 2))}] {progress:.2f}%", end='')

        image_init = cv2.imread(str(image_path))
        stem = image_path.stem
        ext = image_path.suffix

        # Copie de l'original dans augmented/
        cv2.imwrite(str(out_dir / f"{stem}{ext}"), image_init)

        # Génération des variantes
        for k in range(nb_augmentations):
            image = image_init.copy()
            rnd_params = randomize_image(image.shape)
            aug_image = transform_image(image, rnd_params)
            cv2.imwrite(str(out_dir / f"{stem}_{k:02d}{ext}"), aug_image)

    print() 
    print(f"   Images sauvegardées dans : {out_dir}")

print(f"\n Augmentation terminée.")