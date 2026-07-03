import numpy as np
import pandas as pd
from pathlib import Path
import cv2

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


def load_liste_images(resized_dimension=(128,128), category='bottle', type='train', limit_to=None, image_list_path=image_list_path):
    """
    Load a list of images from the specified category and type, resize them, and return both the original and processed images.
    - resized_dimension: tuple specifying the new size (width, height) (default (128,128))
    - category: category of images to load (default 'bottle')
    - type: type of images to load (default 'train')
    - limit_to: maximum number of images to load (default None, which means no limit)
    - image_list_path: path to the CSV file containing image metadata
    Returns:
    - images_originales: numpy array of original images (yet resized and in gray)
    - images: numpy array of processed images (flattened and normalized)
    """
    df = pd.read_csv(image_list_path, dtype={'file':str})

    fichiers = df.loc[(df["category"]==category) & (df['type']==type), ["file", "extension", "type_image"]]

    if limit_to is None or limit_to <= 0:
        nb_images = len(fichiers)
    else:
        nb_images = limit_to

    image_list = []
    i = 0
    for _, f in fichiers[:nb_images].iterrows():
        # display visual progress bar (bar that is growing up to 100%)
        i+= 1
        loading_bar(i, nb_images)

        file_path = current_path.joinpath('data', 'bottle', f["file"] + f["extension"])
        #print(file_path, ":", f["type_image"])
        if f["type_image"]=='C':
            img = cv2.imread(str(file_path), cv2.IMREAD_COLOR)
        else:
            img = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)

        if img is None:
            print(f"Erreur de lecture : {file_path}")
        else:
            
            img_resized = cv2.resize(img, resized_dimension)
            if f["type_image"]=='C':
                img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
            
            image_list.append(img_resized)

    images_originales = np.array(image_list)
    # écrasement des pixels sur une dimension
    images = images_originales.reshape(-1, resized_dimension[0] * resized_dimension[1])

    # rescaling des pixels entre 0 et 1
    images = images.astype("float32") / 255.

    print(f"\rNombre d'images chargées : {len(images)}")

    return images_originales, images