import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from mutinfo.torch.layers import AdditiveGaussianNoise


class MNIST_Classifier(torch.nn.Module):
    """
    MNIST Convolutional classifier.
    """
    
    def __init__(self, sigma: float=0.1):
        super().__init__()
        self.sigma = sigma
        
        # Noise.
        self.dropout = torch.nn.Dropout(0.1)
        self.agn = AdditiveGaussianNoise(sigma, enabled_on_inference=True)
        
        # Activations.
        self.activation = torch.nn.LeakyReLU()
        self.logsoftmax = torch.nn.LogSoftmax(dim=-1)
        
        # Convolutions.
        #torch.nn.utils.parametrizations.spectral_norm()
        self.conv2d_1 = torch.nn.Conv2d(1, 8, kernel_size=3)
        self.conv2d_2 = torch.nn.Conv2d(8, 16, kernel_size=3)
        self.conv2d_3 = torch.nn.Conv2d(16, 32, kernel_size=3)
        self.maxpool2d = torch.nn.MaxPool2d((2,2))
        
        # Dense.
        self.linear_1 = torch.nn.Linear(32, 32)
        self.linear_2 = torch.nn.Linear(32, 10)


    def forward(self, x: torch.tensor, all_layers: bool=False) -> torch.tensor:
        # Convolution №1
        x = self.dropout(x)
        x = self.agn(x)
        x = self.conv2d_1(x)
        x = self.maxpool2d(x)
        layer_1 = self.activation(x)
        
        # Convolution №2
        x = self.dropout(layer_1)
        x = self.agn(x)
        x = self.conv2d_2(x)
        x = self.maxpool2d(x)
        layer_2 = self.activation(x)
        
        # Convolution №3
        x = self.dropout(layer_2)
        x = self.agn(x)
        x = self.conv2d_3(x)
        x = self.maxpool2d(x)
        layer_3 = self.activation(x)
        
        # Dense №1
        x = self.agn(torch.flatten(layer_3, 1))
        x = self.linear_1(x)
        layer_4 = self.activation(x)
        
        # Dense №2
        x = self.agn(layer_4)
        x = self.linear_2(x)
        layer_5 = self.logsoftmax(x)
        
        if all_layers:
            return {
                'layer 1': layer_1,
                'layer 2': layer_2,
                'layer 3': layer_3,
                'layer 4': layer_4,
                'layer 5': layer_5,
                'exp(layer 5)': torch.exp(layer_5),
                #'log(exp(layer 5))': torch.log(torch.exp(layer_5))
            }
        else:
            return layer_5



class CIFAR10_Classifier(torch.nn.Module):
    """
    CIFAR10 Convolutional classifier.
    """
    
    def __init__(self, sigma: float=0.1):
        super().__init__()
        self.sigma = sigma
        
        # Noise.
        self.dropout = torch.nn.Dropout(0.1)
        self.agn = AdditiveGaussianNoise(sigma, enabled_on_inference=True)
        
        # Activations.
        self.activation = torch.nn.LeakyReLU()
        self.logsoftmax = torch.nn.LogSoftmax(dim=-1)
        
        # Convolutions.
        #torch.nn.utils.parametrizations.spectral_norm()
        self.conv2d_1 = torch.nn.Conv2d(3, 8, kernel_size=3, padding='same')
        self.conv2d_2 = torch.nn.Conv2d(8, 16, kernel_size=3, padding='same')
        self.conv2d_3 = torch.nn.Conv2d(16, 32, kernel_size=3, padding='same')
        self.maxpool2d = torch.nn.MaxPool2d((2,2))
        
        # Dense.
        self.linear_1 = torch.nn.Linear(4*4*32, 32)
        self.linear_2 = torch.nn.Linear(32, 10)


    def forward(self, x: torch.tensor, all_layers: bool=False) -> torch.tensor:
        # Convolution №1
        x = self.dropout(x)
        x = self.agn(x)
        x = self.conv2d_1(x)
        x = self.maxpool2d(x)
        layer_1 = self.activation(x)
        
        # Convolution №2
        x = self.dropout(layer_1)
        x = self.agn(x)
        x = self.conv2d_2(x)
        x = self.maxpool2d(x)
        layer_2 = self.activation(x)
        
        # Convolution №3
        x = self.dropout(layer_2)
        x = self.agn(x)
        x = self.conv2d_3(x)
        x = self.maxpool2d(x)
        layer_3 = self.activation(x)
        
        # Dense №1
        x = self.agn(torch.flatten(layer_3, 1))
        x = self.linear_1(x)
        layer_4 = self.activation(x)
        
        # Dense №2
        x = self.agn(layer_4)
        x = self.linear_2(x)
        layer_5 = self.logsoftmax(x)
        
        if all_layers:
            return {
                'layer 1': layer_1,
                'layer 2': layer_2,
                'layer 3': layer_3,
                'layer 4': layer_4,
                'layer 5': layer_5,
                'exp(layer 5)': torch.exp(layer_5),
                #'log(exp(layer 5))': torch.log(torch.exp(layer_5))
            }
        else:
            return layer_5



def evaluate_classifier(classifier, dataloader, classifier_loss, device) -> (float, float):
    """
    Calculate metrics for the classifier.
    """
    
    # Exit training mode.
    was_in_training = classifier.training
    classifier.eval()
    
    # Targets and predictions.
    y_all = []
    y_pred_all = []
    
    with torch.no_grad():
        avg_loss = 0.0
        total_samples = 0
        for index, batch in enumerate(dataloader):
            x, y = batch
            batch_size = x.shape[0]
            
            y_pred = classifier(x.to(device))
            _loss = classifier_loss(y_pred, y.to(device))

            avg_loss += _loss.item() * batch_size
            total_samples += batch_size

            y_pred_all.append(np.exp(y_pred.detach().cpu().numpy()))
            y_all.append(y.detach().cpu().numpy())
            
        avg_loss /= total_samples
        
    # ROC AUC
    y_pred_all = np.vstack(y_pred_all)
    y_all = np.concatenate(y_all)
    roc_auc = roc_auc_score(y_all, y_pred_all, multi_class='ovo')
        
    # Return to the original mode.
    classifier.train(was_in_training)
    
    return avg_loss, roc_auc