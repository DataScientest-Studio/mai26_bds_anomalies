import torch
from torch.utils.data import DataLoader

from dataloader import AnomalyDataset
from classifier_models import CustomModel

import matplotlib.pyplot as plt

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

    model = CustomModel(4, device)

    train_dataset = AnomalyDataset(category=category, train=True, transform=model.preprocess)
    test_dataset = AnomalyDataset(category=category, train=False, transform=model.preprocess)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    print(model.test_criterion(train_dataloader))

    losses = model.train(train_dataloader, test_dataloader, epochs=50)

    plt.figure(figsize=(16,6))
    plt.plot(losses['train'], label='Train Loss')
    plt.plot(losses['test'], label='Test Loss')
    plt.legend()
    plt.title('Loss Over Epochs')
    plt.xlabel('Loss')
    plt.show();