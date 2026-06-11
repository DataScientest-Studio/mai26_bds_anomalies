from pathlib import Path
from PIL import Image
import pandas as pd
from dotenv import load_dotenv
import os

load_dotenv()

root_path = Path(os.getenv("PATH_DATASET"))
csv_file = "image_list.csv"

def get_image_characteristics(image_file):
    with Image.open(image_file) as img:
        largeur, hauteur = img.size
        mode = img.mode

        # Nombre de canaux couleur
        nb_canaux = len(img.getbands())

        # Interprétation simple
        if mode in ["L", "1"]:
            type_image = "G" # grey / niveau de gris ou noir&blanc
        else:
            type_image = "C" # couleur

        resultat = {
            "width": largeur,
            "height": hauteur,
            "mode": mode,
            "bands": nb_canaux,
            "type_image": type_image
        }
        return resultat

def get_mask(qual_path_gt, image_file):
    f_gt = image_file.stem + "_mask" + image_file.suffix
    assert qual_path_gt.joinpath(f_gt).is_file(), Exception("File not found in ground_truth: "+f_gt)
    return qual_path_gt.joinpath(f_gt)

def get_image_details(cat, qual, img, qual_path_gt=None):
    assert img.is_file(), Exception("Not a file: "+str(img))
    assert img.suffix in [".png"], Exception("Unsupported file type: "+img.suffix)
    img_info = get_image_characteristics(img)
    img_info["category"]=cat.name
    img_info["type"]=qual.parent.name
    img_info["quality"]=qual.name
    img_info["file"]=str(img.stem)
    img_info["extension"]=img.suffix
    if qual.name != "good":
        mask = get_mask(qual_path_gt, img)
        assert mask.suffix in [".png"], Exception("Unsupported file type: "+mask.suffix)
        mask_info = get_image_characteristics(mask)
        assert img_info["width"] == mask_info["width"] and img_info["height"] == mask_info["height"], Exception("Image and mask dimensions do not match")
        img_info["mask"]=True
        img_info["mask_bands"]=mask_info["bands"]
        img_info["mask_mode"]=mask_info["mode"]
        img_info["mask_type"]=mask_info["type_image"]
    else:
        img_info["mask"]=False
    return img_info


images=pd.DataFrame(columns=["category", "type", "quality", "file", "extension", 
                             "width", "height", "mode", "bands", "type_image", 
                             "mask", "mask_bands", "mask_mode", "mask_type"])
for cat in root_path.iterdir():

    if cat.is_dir():
        print("----")
        print("Category:", cat.name)

        # ground_truth + test
        for qual in cat.joinpath("test").iterdir():
            print("TEST - Quality:", qual.name)
            qual_path_gt = cat.joinpath("ground_truth", qual.name)

            # Vérification que le dossier et les fichiers existent aussi dans ground_truth
            if qual.name != "good":
                assert qual_path_gt.is_dir(), Exception("Dir not found in ground_truth")

            # Parcours des images
            for img in qual.iterdir():
                images.loc[len(images)] = get_image_details(cat, qual, img, qual_path_gt)

        # train
        for qual in cat.joinpath("train").iterdir():
            print("TRAIN - Quality:", qual.name)
            assert qual.name=="good", Exception("Train quality shoud be good")
            qual_path_gt = cat.joinpath("ground_truth", qual.name)

            # Parcours des images
            for img in qual.iterdir():
                images.loc[len(images)] = get_image_details(cat, qual, img)

# save image list in csv
images.to_csv(csv_file, index=False)