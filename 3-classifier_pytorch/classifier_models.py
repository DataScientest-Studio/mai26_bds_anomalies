import torch
import torchvision.models as models
from tqdm.auto import tqdm
import numpy as np

initial_lr = 1e-1
reduce_lr = {
    'mode': 'min',
    'factor': 0.1, 
    'patience': 5,
}
epochs=10

class CustomModel():

    def __init__(self, num_classes, device):
        """ Create a pre-trained EfficientNet B0 model and freezes all layers
        Parameters:
        - num_classes: number of output classes
        - device: the device to use (cpu or cuda)

        Initializes:
        - device: the device to use (cpu or cuda)
        - model: the pre-trained EfficientNet B0 model
        - preprocess: a function to preprocess the input image
        - criterion: the loss function
        - optimizer: the Adam optimizer
        - scheduler: the ReduceLROnPlateau scheduler
        """
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        model = models.efficientnet_b0(weights = weights)

        for param in model.parameters():
            param.requires_grad = False

        model.classifier[-1] = torch.nn.Linear(
            model.classifier[-1].in_features, num_classes, 
        )
        self.model = model
        self.model.to(device)
        self.device = device

        self.preprocess = weights.transforms()
        self.criterion = torch.nn.CrossEntropyLoss()

        self.optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, **reduce_lr)

    def test_criterion(self, dataloader):
        X_batch, y_batch = next(iter(dataloader))
        
        X_batch = self.preprocess(X_batch.to(self.device))
        y_pred = self.model( X_batch )
        
        return self.criterion(y_pred, y_batch.to(self.device))
    
    def train(self, dataloader_train, dataloader_test, epochs=epochs):
        """ Trains the model on a given dataset and evaluates it on
        Parameters:
        - dataloader_train: DataLoader object containing the training data
        - dataloader_test: DataLoader object containing the test data
        - epochs: Number of epochs to train the model
        Returns:
        - losses: Dictionary with the keys:
           - train: List of training losses
           - test : List of test losses
        """
        losses = {'train': [], 'test' :[]}

        for epoch in range(epochs):
            # Entrainement
            self.model.train()
            
            loss_train=[]

            progress_bar=tqdm( 
                dataloader_train, 
                desc = f"Train: Epoch {epoch+1}/{epochs}", leave=False, 
            )
            for batch in progress_bar:
                X, y = [b.to(self.device) for b in batch]
                self.model.zero_grad()

                y_hat = self.model(self.preprocess(X))

                loss_value = self.criterion(y_hat, y)
                loss_value.backward()

                loss_train.append(loss_value.item())
                self.optimizer.step()
            
            loss_train_value = np.mean(loss_train)
            losses['train'].append( loss_train_value )

            # Evaluation
            loss_test = []
            self.model.eval()
            
            progress_bar=tqdm( 
                dataloader_test, 
                desc = f"Eval: Epoch {epoch+1}/{epochs}", leave=False, 
            )
            for batch in progress_bar:
                X, y = [b.to(self.device) for b in batch]
                with torch.no_grad():
                    y_hat = self.model(self.preprocess(X))
                
                loss_value = self.criterion(y_hat, y)
                loss_test.append(loss_value.item())
            
            loss_test_value = np.mean(loss_test)
            losses['test'].append( loss_test_value )

            self.scheduler.step( loss_test_value )
            print(f"Epoch {epoch+1}/{epochs}: " + 
                  f"Train Loss={loss_train_value} - " + 
                  f"Test Loss: {loss_test_value}"
            )

        return losses