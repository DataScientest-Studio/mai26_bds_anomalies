from pathlib import Path
from PIL import Image
import pandas as pd
from dotenv import load_dotenv
import os

load_dotenv('sample.env')

root_path = Path(os.getenv("PATH_DATASET_RAD"))

if (root_path / root_path.name).exists():
    root_path = root_path / root_path.name

print("\n===== DEBUG START =====")
print("ROOT PATH:", root_path)
print("EXISTS:", root_path.exists())

if not root_path.exists():
    raise ValueError("Dataset path does not exist! Check .env")

csv_file = "image_list_RAD.csv"

for cat in root_path.iterdir():
    if not cat.is_dir():
        continue
    gt_root = cat / "ground_truth"
    if gt_root.exists():
        for qual in gt_root.iterdir():
            if qual.name != "good":
                print("GT files:", sorted(qual.iterdir())[:10])
                break
    break

def get_image_characteristics(image_file):
    with Image.open(image_file) as img:
        width, height = img.size
        mode = img.mode
        bands = len(img.getbands())

        return {
            "width": width,
            "height": height,
            "mode": mode,
            "bands": bands,
            "type_image": "G" if mode in ["L", "1"] else "C"
        }

def get_mask(gt_dir, image_file):
    f_gt = image_file.name
    mask_path = gt_dir / f_gt
    if not mask_path.is_file():
        return None  # pas de mask pour cette image
    return mask_path

def get_image_details(cat, split, qual, img):
    assert img.is_file(), Exception("Not a file: " + str(img))
    assert img.suffix in [".png"], Exception("Unsupported file type: " + img.suffix)

    img_info = get_image_characteristics(img)
    img_info["category"] = cat.name
    img_info["type"] = split
    img_info["quality"] = qual.name
    img_info["file"] = img.stem
    img_info["extension"] = img.suffix

    gt_dir = cat / "ground_truth" / qual.name

    if qual.name != "good":
        mask = get_mask(gt_dir, img)
        if mask is not None:
            mask_info = get_image_characteristics(mask)
            assert img_info["width"] == mask_info["width"] and img_info["height"] == mask_info["height"], \
                Exception("Image and mask dimensions do not match")
            img_info["mask"] = True
            img_info["mask_bands"] = mask_info["bands"]
            img_info["mask_mode"] = mask_info["mode"]
            img_info["mask_type"] = mask_info["type_image"]
        else:
            print("No mask found for:", img.name)
            img_info["mask"] = False
    else:
        img_info["mask"] = False

    return img_info



# LOOP DATASET

rows = []

print("\nCategories found:")
print([d.name for d in root_path.iterdir() if d.is_dir()])

for cat in root_path.iterdir():

    if not cat.is_dir():
        continue

    print("\n---- CATEGORY:", cat.name)

    test_dir = cat / "test"
    train_dir = cat / "train"

    print("TEST EXISTS:", test_dir.exists())
    print("TRAIN EXISTS:", train_dir.exists())

    # TEST
    for qual in test_dir.iterdir():
        print("TEST -", qual.name)

        if qual.name != "good":
            gt_dir = cat / "ground_truth" / qual.name
            assert gt_dir.is_dir(), Exception("Dir not found in ground_truth: " + str(gt_dir))

        for img in qual.iterdir():
            rows.append(get_image_details(cat, "test", qual, img))

  
    for qual in train_dir.iterdir():
        print("TRAIN -", qual.name)
        assert qual.name == "good", Exception("Train quality should be good")

        for img in qual.iterdir():
            rows.append(get_image_details(cat, "train", qual, img))


images = pd.DataFrame(rows, columns=["category", "type", "quality", "file", "extension",
                                     "width", "height", "mode", "bands", "type_image",
                                     "mask", "mask_bands", "mask_mode", "mask_type"])

print("\n===== FINAL STATS =====")
print("Total images processed:", len(images))
print("Saving CSV to:", csv_file)

images.to_csv(csv_file, index=False)

print("CSV CREATED:", Path(csv_file).exists())
