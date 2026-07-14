import cv2
from pathlib import Path
from dotenv import load_dotenv
from os import getenv
import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf

load_dotenv()
root_path = Path(getenv("PATH_DATASET")) / "screw"
new_path = Path(getenv("PATH_DATASET")) / "screw_preprocessed"

def get_screw_contour(image):
    # Je crée un masque en noir et blanc
    mask = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    mask = cv2.threshold(mask, 128., 255., type=cv2.THRESH_BINARY_INV)[1]
    mask = mask.astype(np.uint8)

    # Je calcule les contours et je sélectionne le plus grand (en aire)
    contours, _ = cv2.findContours(mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    contour_areas=list(map(cv2.contourArea, contours))
    contour = contours[np.argmax(contour_areas)]

    return contour

def align_horizontally(image, mask=None):
    contour = get_screw_contour(image)

    [vx, vy, x0, y0] = cv2.fitLine(contour, cv2.DIST_L2,0,0.01,0.01)
    angle = np.degrees(np.arctan2(vy, vx))[0]

    image, mask= rotate_image(image,angle, mask)

    return image, mask

def rotate_image(image, angle, mask=None):
    center=(image.shape[1]//2, image.shape[0]//2)
    R = cv2.getRotationMatrix2D(center, angle, scale=1)

    cos = abs(R[0, 0])
    sin = abs(R[0, 1])

    new_width = int(np.ceil(image.shape[0] * sin + image.shape[1] * cos))
    new_height = int(np.ceil(image.shape[0] * cos + image.shape[1] * sin))

    # Décalage pour recentrer l'image dans le nouveau canevas
    R[0, 2] += new_width / 2 - center[0]
    R[1, 2] += new_height / 2 - center[1]

    image = cv2.warpAffine(
        image, R, #(image.shape[1], image.shape[0]),
        (new_width, new_height),
        borderMode=cv2.BORDER_REPLICATE,
    )
    #print("Rotation :", angle)
    if mask is not None:
        mask = cv2.warpAffine(
            mask, R, #(image.shape[1], image.shape[0]),
            (new_width, new_height),
            borderMode=cv2.BORDER_REPLICATE,
        )
    
    return image, mask

def center_crop_screw(image, dimension_shape, mask=None):
    contour = get_screw_contour(image)

    [x,y,wr,hr] = cv2.boundingRect(contour)

    h, w = dimension_shape[:2]
    translation_x = w//2 - (x + wr//2)
    translation_y = h//2 - (y + hr//2)

    matrix = np.float32([
        [1, 0, translation_x],
        [0, 1, translation_y]
    ])

    image = cv2.warpAffine(
        image,
        matrix,
        (image.shape[1], image.shape[0]),
        borderMode=cv2.BORDER_REFLECT,
    )

    image = image[:h, :w]

    if mask is not None:
        mask = cv2.warpAffine(
            mask, 
            matrix,
            (image.shape[1], image.shape[0]),
            borderMode=cv2.BORDER_REFLECT,
        )
        mask = mask[:h, :w]

    return image, mask

def measure_max_height(image):
    contour = get_screw_contour(image)
    _,_,_,h = cv2.boundingRect(contour) # returns x, y, w, h

    return h

def detect_head_side(image):
    left_image = image[:,:image.shape[1]//2,...]
    right_image = image[:,image.shape[1]//2:,...]

    h_left=measure_max_height(left_image)
    h_right=measure_max_height(right_image)

    #print(f"hauteur gauche={h_left}, hauteur droite={h_right}")

    if h_left >= h_right:
        return 0 # left
    else:
        return 1 # right

def preprocess_screw(image, mask=None):
    image = image.astype(np.uint8)
    initial_shape = image.shape

    image, mask = align_horizontally(image, mask)

    side = detect_head_side(image)
    if side > 0:
        image, mask=rotate_image(image, 180, mask)

    image, mask = center_crop_screw(image, initial_shape, mask)
    
    if mask is None:
        return image
    else:
        return image, mask

def preprocess_screw_tf(image):
    processed_image = tf.numpy_function(
        func=preprocess_screw,
        inp=[image],
        Tout=tf.uint8,
    )

    # numpy_function fait perdre l'information de shape
    processed_image.set_shape(image.shape)

    return processed_image

def preprocess_screw_batch(image_batch):
    return tf.map_fn(
        preprocess_screw_tf,
        image_batch,
        fn_output_signature=tf.uint8,
    )


def __main__():
    """
    image_number=np.random.randint(0,319)
    image_number = 114
    print(f"{image_number:03d}.png")

    image_path = Path("/home/win100/Documents/projet_liora/mvtec_anomaly_detection/screw/train/good/")
    image = cv2.imread(
        image_path / f"{image_number:03d}.png"
    )

    image_modified = preprocess_screw(image)

    plt.imshow(image_modified)
    # Dessin du contour
    # plt.plot(
    #     contour[:, 0, 0],
    #     contour[:, 0, 1],
    #     label="Contour"
    # )

    plt.axis('off')
    plt.show();
    """

    # images "good"
    image_paths=[
        {
            'image_path': Path(root_path / "train" / "good"), 
            'new_image_path': Path(new_path / "train" / "good"), 
            'mask_path': None, 
            'new_mask_path': None, 
        },
        {
            'image_path': Path(root_path / "test" / "good"), 
            'new_image_path': Path(new_path / "test" / "good"), 
            'mask_path': None, 
            'new_mask_path': None, 
        },
    ]
    image_paths[0]['new_image_path'].mkdir(parents=True, exist_ok=True)
    image_paths[1]['new_image_path'].mkdir(parents=True, exist_ok=True)

    # recherche de toutes les catégories d'anomalies (avec masque dans ground_truth)
    test_path = Path(root_path / "test")
    for d in test_path.iterdir():
        if d.is_dir() and d.name!="good":
            anomaly_dir = {
                'image_path': d, 
                'new_image_path': Path(new_path / "test" / d.name), 
                'mask_path': Path(root_path / "ground_truth" / d.name), 
                'new_mask_path': Path(new_path / "ground_truth" / d.name), 
            }
            anomaly_dir['new_image_path'].mkdir(parents=True, exist_ok=True)
            anomaly_dir['new_mask_path'].mkdir(parents=True, exist_ok=True)

            image_paths.append(anomaly_dir)

    for dir in image_paths:
        for f in dir["image_path"].iterdir():
            has_mask = (dir["mask_path"]!=None)
            if f.is_file() and f.suffix in ['.png', '.jpg', '.jpeg']:
                image = cv2.imread(f)

                mask=None
                if has_mask:
                    print(f"Processing { f.parent.parent.name }/{ f.parent.name }/{f.name} (file and mask)")
                    mask_file_name = f.stem + "_mask.png"
                    mask = cv2.imread(dir["mask_path"] / mask_file_name)

                    image_modified, mask_modified = preprocess_screw(image, mask)

                    cv2.imwrite(dir["new_image_path"] / f.name, image_modified)
                    cv2.imwrite(dir["new_mask_path"] / mask_file_name, mask_modified)
                    print(f"Saved { f.parent.parent.name }/{ f.parent.name }/{f.name} (file and mask)")
                else:
                    print(f"Processing { f.parent.parent.name }/{ f.parent.name }/{f.name} (file only)")
                    image_modified = preprocess_screw(image)

                    cv2.imwrite(dir["new_image_path"] / f.name, image_modified)
                    print(f"Saved { f.parent.parent.name }/{ f.parent.name }/{f.name} (file only)")

#__main__()