import torch
from torch.utils.data import DataLoader

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

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'

for category in categories:

    train_dataset = AnomalyDataset(category=category, train=True, 
                                   binary=True)

    model = CustomModel(num_classes=len(train_dataset.classes), device=device)
    train_dataset.transform = model.preprocess

    test_dataset = AnomalyDataset(category=category, train=False, 
                                  transform=model.preprocess, binary=True)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    #print(model.test_criterion(train_dataloader))
    model.unfreeze_last_n_layers(8)

    print(f"Classes : {train_dataset.classes}")
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
