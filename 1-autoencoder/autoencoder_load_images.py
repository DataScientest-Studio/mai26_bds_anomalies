import numpy as np
from pathlib import Path
import cv2
import random

current_path = Path(__file__).parent.resolve()
image_list_path = current_path.joinpath('image_list.csv')

def loading_bar(ind, total, hide_if_100=True, bar_length=40):
    """
    Display a loading bar in the console.
    - ind: current index (1-based)
    - total: total number of items
    - hide_if_100: if True, hide the bar when progress reaches 100%
    - bar_length: length of the loading bar in characters
    """
    progress = ind / total
    
    if hide_if_100 and progress >= 1.:
        print(f"\r{' '*200}", end="")  # Clear the progress bar line
    else:
        filled = int(bar_length * progress)
        bar = '█' * filled + '░' * (bar_length - filled)
        print(f"\rProgression : [{bar}] {progress*100:.1f}%", end="")


def load_liste_images(images_path, resized_dimension=(128,128), category='bottle', type='train', quality='good', limit_to=None):
    """
    Load a list of images from the specified category and type, resize them, and return both the original and processed images.
    - resized_dimension: tuple specifying the new size (width, height) (default (128,128))
    - category: category of images to load (default 'bottle')
    - quality: quality of images to load (default 'good')
    - type: type of images to load (default 'train')
    - limit_to: maximum number of images to load (default None, which means no limit)
    Returns:
    - images_originales: numpy array of original images (yet resized and in gray)
    - images: numpy array of processed images (flattened and normalized)
    """

    images_path = images_path.joinpath(category, type, quality)
    # check if 'augmented' directory exists, if not keep images_path as is, otherwise use 'augmented' directory
    augmented_path = images_path.joinpath('augmented')
    if augmented_path.exists():
        images_path = augmented_path

    # count images in images_path
    fichiers = []
    for f in images_path.glob("*"):
        if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            fichiers.append(f)
    random.shuffle(fichiers)

    if limit_to is None or limit_to <= 0:
        nb_images = len(fichiers)
    else:
        nb_images = limit_to

    image_list = []
    for i, f in enumerate(fichiers[:nb_images]):
        # display visual progress bar (bar that is growing up to 100%)
        loading_bar(i+1, nb_images)

        file_path = images_path.joinpath(fichiers[i])
        #print(file_path, ":", f["type_image"])
        img = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)

        if img is None:
            print(f"Erreur de lecture : {file_path}")
        else:
            
            img_resized = cv2.resize(img, resized_dimension)
            
            image_list.append(img_resized)

    images_originales = np.array(image_list)
    # écrasement des pixels sur une dimension
    images = images_originales.reshape(-1, resized_dimension[0] * resized_dimension[1])

    # rescaling des pixels entre 0 et 1
    images = images.astype("float32") / 255.

    print(f"\rNombre d'images chargées : {len(images)}")

    return images_originales, images