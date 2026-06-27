"""
Pipeline d'enrichissement de données - Détection d'anomalies industrielles
Contexte : Images RAD  (PNG/JPEG)
Approche : Non supervisée, on enrichit uniquement les images "normales"
"""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import Generator
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# 1. CONFIGURATION CENTRALE

CONFIG = {
    "input_dir": "./data/RAD-dataset",           # Dossier source images (PNG/JPEG)
    "output_dir": "./data/augmented",    # Dossier de sortie
    "image_size": (256, 256),            # Taille cible — adapter selon tes images
    "batch_size": 32,                    # Faible pour économiser la RAM (<16 Go)
    "augmentations_per_image": 5,        # Nombre de variantes générées par image
    "save_format": "png",                # Format de sortie
    "num_workers": 2,                    # Threads DataLoader (limité pour la RAM)
    "seed": 42,
}


# 2. PIPELINES D'AUGMENTATION

def get_geometric_transform(image_size: tuple) -> A.Compose:
    """
    Augmentations géométriques.
    Adaptées aux images industrielles : on évite les distorsions trop extrêmes
    qui effaceraient les patterns normaux de la pièce.
    """
    return A.Compose([
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.1,
            rotate_limit=15,
            border_mode=cv2.BORDER_REFLECT_101,
            p=0.6
        ),
        A.RandomCrop(
            height=int(image_size[0] * 0.85),
            width=int(image_size[1] * 0.85),
            p=0.4
        ),
        A.Resize(height=image_size[0], width=image_size[1]),
        A.Perspective(scale=(0.02, 0.05), p=0.3),
        A.ElasticTransform(
            alpha=30,
            sigma=5,
            p=0.2   # Faible proba : préserve la structure industrielle
        ),
    ])


def get_photometric_transform() -> A.Compose:
    """
    Augmentations photométriques.
    Simule les variations réelles d'acquisition : éclairage, capteur, compression.
    """
    return A.Compose([
        # Bruit capteur
        A.GaussNoise(var_limit=(5.0, 25.0), p=0.5),
        A.ISONoise(color_shift=(0.01, 0.03), intensity=(0.05, 0.15), p=0.3),

        # Flou / mise au point imparfaite
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5)),
            A.MotionBlur(blur_limit=5),
            A.MedianBlur(blur_limit=3),
        ], p=0.4),

        # Luminosité et contraste
        A.RandomBrightnessContrast(
            brightness_limit=0.15,
            contrast_limit=0.15,
            p=0.6
        ),
        A.RandomGamma(gamma_limit=(85, 115), p=0.3),

        # Amélioration locale du contraste (CLAHE) — utile images industrielles
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.4),

        # Compression JPEG simulée (artefacts réseau/stockage)
        A.ImageCompression(quality_lower=75, quality_upper=95, p=0.2),

        # Teintes légères (si images couleur)
        A.HueSaturationValue(
            hue_shift_limit=5,
            sat_shift_limit=15,
            val_shift_limit=10,
            p=0.3
        ),
    ])


def get_combined_transform(image_size: tuple) -> A.Compose:
    """
    Pipeline combiné : géométrique + photométrique.
    Ordre important : géométrique EN PREMIER, photométrique ensuite.
    """
    geo = get_geometric_transform(image_size)
    photo = get_photometric_transform()

    return A.Compose(
        geo.transforms + photo.transforms,
        additional_targets={}
    )


def get_normalization_transform(image_size: tuple) -> A.Compose:
    """
    Normalisation finale pour entrée dans un modèle deep learning.
    Valeurs ImageNet classiques — à adapter si tes images sont très différentes.
    """
    return A.Compose([
        A.Resize(height=image_size[0], width=image_size[1]),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])



# 3. DATASET AVEC CHARGEMENT EN STREAMING

class IndustrialImageDataset(Dataset):
    """
    Dataset économe en RAM : charge les images à la volée (pas tout en mémoire).
    Compatible avec DataLoader multi-worker.
    """

    SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}

    def __init__(self, root_dir: str, image_size: tuple = (256, 256), transform=None):
        self.root_dir = Path(root_dir)
        self.image_size = image_size
        self.transform = transform
        self.image_paths = self._scan_images()
        logger.info(f"Dataset chargé : {len(self.image_paths)} images trouvées dans {root_dir}")

    def _scan_images(self) -> list:
        paths = [
            p for p in self.root_dir.rglob("*")
            if p.suffix.lower() in self.SUPPORTED_EXTENSIONS
        ]
        if not paths:
            raise FileNotFoundError(f"Aucune image trouvée dans {self.root_dir}")
        return sorted(paths)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict:
        img_path = self.image_paths[idx]
        image = self._load_image(img_path)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        return {
            "image": image,
            "path": str(img_path),
            "stem": img_path.stem,
        }

    def _load_image(self, path: Path) -> np.ndarray:
        """Charge et redimensionne une image de façon robuste."""
        try:
            img = cv2.imread(str(path))
            if img is None:
                raise ValueError(f"cv2 ne peut pas lire : {path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.image_size)
            return img
        except Exception:
            # Fallback PIL pour formats exotiques
            img = Image.open(path).convert("RGB")
            img = img.resize(self.image_size, Image.LANCZOS)
            return np.array(img)


# 4. GÉNÉRATEUR DE PATCHES (optionnel, mais utile pour detection anomalie)


def extract_patches(
    image: np.ndarray,
    patch_size: int = 64,
    stride: int = 32
) -> Generator[np.ndarray, None, None]:
    """

    Args:
        image     : np.ndarray (H, W, C)
        patch_size: taille du patch carré
        stride    : décalage entre deux patches (stride < patch_size = chevauchement)

    Yields:
        patch np.ndarray (patch_size, patch_size, C)
    """
    h, w = image.shape[:2]
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            yield image[y:y + patch_size, x:x + patch_size]


# 5. PIPELINE PRINCIPAL D'ENRICHISSEMENT

class AugmentationPipeline:
    """
    Orchestre le chargement, l'augmentation et la sauvegarde.
    """

    def __init__(self, config: dict):
        self.config = config
        self.output_dir = Path(config["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.transform = get_combined_transform(config["image_size"])
        np.random.seed(config["seed"])

    def run(self):
        dataset = IndustrialImageDataset(
            root_dir=self.config["input_dir"],
            image_size=self.config["image_size"],
            transform=None,   # On applique la transform manuellement (N fois)
        )

        loader = DataLoader(
            dataset,
            batch_size=self.config["batch_size"],
            shuffle=False,
            num_workers=self.config["num_workers"],
            pin_memory=False,   # Désactivé : économise la RAM
        )

        total_saved = 0
        n_aug = self.config["augmentations_per_image"]

        for batch_idx, batch in enumerate(loader):
            paths = batch["path"]
            stems = batch["stem"]

            # Récupère les images brutes depuis le disque (pas via DataLoader tensor)
            for i, (img_path, stem) in enumerate(zip(paths, stems)):
                raw_img = cv2.imread(img_path)
                if raw_img is None:
                    logger.warning(f"Image illisible ignorée : {img_path}")
                    continue
                raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
                raw_img = cv2.resize(raw_img, self.config["image_size"])

                # Sauvegarde de l'image originale
                self._save_image(raw_img, stem, variant=0)
                total_saved += 1

                # Génération des N variantes augmentées
                for aug_idx in range(1, n_aug + 1):
                    augmented = self.transform(image=raw_img)
                    aug_img = augmented["image"]
                    self._save_image(aug_img, stem, variant=aug_idx)
                    total_saved += 1

            if (batch_idx + 1) % 10 == 0:
                logger.info(
                    f"Batch {batch_idx + 1}/{len(loader)} traité — "
                    f"{total_saved} images sauvegardées"
                )

        logger.info(f"\n✅ Enrichissement terminé : {total_saved} images dans {self.output_dir}")
        return total_saved

    def _save_image(self, image: np.ndarray, stem: str, variant: int):
        filename = f"{stem}_aug{variant:02d}.{self.config['save_format']}"
        output_path = self.output_dir / filename
        img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_path), img_bgr)


# 6. DATALOADER PRÊT POUR L'ENTRAÎNEMENT

def get_training_dataloader(
    augmented_dir: str,
    image_size: tuple = (256, 256),
    batch_size: int = 32,
    num_workers: int = 2,
) -> DataLoader:
    """
    DataLoader final avec normalisation — à brancher directement sur ton modèle
    (Autoencoder, PatchCore, PADIM, etc.).
    """
    transform = get_normalization_transform(image_size)
    dataset = IndustrialImageDataset(
        root_dir=augmented_dir,
        image_size=image_size,
        transform=transform,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
    )


# 7. POINT D'ENTRÉE

if __name__ == "__main__":
    # ── Étape 1 : Enrichissement et sauvegarde sur disque
    pipeline = AugmentationPipeline(CONFIG)
    pipeline.run()

    # ── Étape 2 : DataLoader prêt pour l'entraînement
    train_loader = get_training_dataloader(
        augmented_dir=CONFIG["output_dir"],
        image_size=CONFIG["image_size"],
        batch_size=CONFIG["batch_size"],
        num_workers=CONFIG["num_workers"],
    )

    # ── Vérification rapide
    batch = next(iter(train_loader))
    logger.info(f"Shape d'un batch : {batch['image'].shape}")
    logger.info(f"Dtype : {batch['image'].dtype}")
    logger.info("Pipeline prêt pour l'entraînement ✅")