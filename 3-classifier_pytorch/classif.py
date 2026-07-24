import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from dataloader import AnomalyDataset
from classifier_models import CustomModel

import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

##########################################################
###                      SETTINGS                      ###
##########################################################

#categories = ['bottle', 'cable', 'capsule', 'carpet', 'grid',
#    'hazelnut', 'leather', 'metal_nut', 'pill', 'screw_preprocessed',
#    'tile', 'toothbrush', 'transistor', 'wood', 'zipper', 
#    'metal_plate']
categories = ['bottle']

batch_size = 8


##########################################################
###                      PROGRAM                       ###
##########################################################

def train_transform(preprocess):
    return transforms.Compose([
        # Augmentations légères: à adapter par catégorie si certaines symétries
        # ne sont pas physiquement plausibles pour tes pièces.
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10),
        transforms.RandomAffine(degrees=0, translate=(0.03, 0.03), scale=(0.95, 1.05)),
        preprocess,
    ])

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'

for category in categories:

    train_dataset = AnomalyDataset(category=category, train=True, 
                                   binary=True)

    model = CustomModel(num_classes=len(train_dataset.classes), device=device)
    train_dataset.transform = train_transform(model.preprocess)

    test_dataset = AnomalyDataset(category=category, train=False, 
                                  transform=model.preprocess, binary=True)

    train_sampler = train_dataset.make_weighted_sampler()
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    #print(model.test_criterion(train_dataloader))
    model.unfreeze_last_n_layers(8)

    print(f"Classes : {train_dataset.classes}")
    print(f"Train class counts : {train_dataset.class_counts()}")
    losses = model.train(train_dataloader, test_dataloader, epochs=10)

    loss, y_true, y_pred = model.evaluate(test_dataloader)
    print(f"Loss = {loss}")
    print("Confusion matrix :")
    print(confusion_matrix(y_true, y_pred))
    print("Classification report :")
    print(classification_report(y_true, y_pred, zero_division=np.nan))

    plt.figure(figsize=(16,6))
    plt.plot(losses['train'], label='Train Loss')
    plt.plot(losses['test'], label='Test Loss')
    plt.legend()
    plt.title('Loss Over Epochs')
    plt.xlabel('Loss')
    plt.show();
