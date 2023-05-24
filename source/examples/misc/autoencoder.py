import torch
from tqdm import tqdm
from mutinfo.torch.layers import AdditiveGaussianNoise


class ConvEncoder(torch.nn.Module):
    """
    Convolutional encoder.
    
    Parameters
    ----------
    latent_dim : int
        Latent representation dimension.
    """
    
    def __init__(self, latent_dim: int):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Noise.
        self.dropout = torch.nn.Dropout(0.1)
        
        # Activations.
        self.activation = torch.nn.LeakyReLU()
        self.sigmoid = torch.nn.Sigmoid()
        
        # Convolutions.
        self.conv2d_1 = torch.nn.Conv2d(1, 8, kernel_size=3, padding='same')
        self.conv2d_2 = torch.nn.Conv2d(8, 16, kernel_size=3, padding='same')
        self.conv2d_3 = torch.nn.Conv2d(16, 32, kernel_size=3, padding='same')
        
        self.maxpool2d = torch.nn.MaxPool2d((2,2))
        
        # Dense.
        self.linear_1 = torch.nn.Linear(288, 128)
        self.linear_2 = torch.nn.Linear(128, self.latent_dim)
        
        
    def forward(self, x: torch.tensor) -> torch.tensor:
        # Convolution №1
        x = self.dropout(x)
        x = self.conv2d_1(x)
        x = self.maxpool2d(x)
        layer_1 = self.activation(x)
        
        # Convolution №2
        x = self.dropout(layer_1)
        x = self.conv2d_2(x)
        x = self.maxpool2d(x)
        layer_2 = self.activation(x)
        
        # Convolution №3
        x = self.dropout(layer_2)
        x = self.conv2d_3(x)
        x = self.maxpool2d(x)
        layer_3 = self.activation(x)
        
        # Dense №1
        x = torch.flatten(layer_3, 1)
        x = self.linear_1(x)
        layer_4 = self.activation(x)
        
        # Dense №2
        x = self.linear_2(layer_4)
        layer_5 = self.sigmoid(x)
        
        return layer_5



class ConvDecoder(torch.nn.Module):
    """
    Convolutional decoder.
    
    Parameters
    ----------
    latent_dim : int
        Latent representation dimension.
    """
        
    def __init__(self, latent_dim: int):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Activations.
        self.activation = torch.nn.LeakyReLU()
        self.sigmoid = torch.nn.Sigmoid()
        
        # Convolutions.
        self.conv2d_1 = torch.nn.Conv2d(32, 16, kernel_size=3, padding='same')
        self.conv2d_2 = torch.nn.Conv2d(16, 8, kernel_size=3, padding='same')
        self.conv2d_3 = torch.nn.Conv2d(8, 1, kernel_size=3, padding='same')
        
        self.upsample = torch.nn.Upsample(scale_factor=2)
        
        # Dense.
        self.linear_1 = torch.nn.Linear(latent_dim, 128)
        self.linear_2 = torch.nn.Linear(128, 1568)
        
        
    def forward(self, x: torch.tensor) -> torch.tensor:
        # Dense №1
        x = self.linear_1(x)
        layer_1 = self.activation(x)
        
        # Dense №2
        x = self.linear_2(layer_1)
        layer_2 = self.activation(x)
        
        # Convolution №1
        x = torch.reshape(layer_2, (-1, 32, 7, 7))
        x = self.conv2d_1(x)
        x = self.upsample(x)
        layer_3 = self.activation(x)
        
        # Convolution №2
        x = self.conv2d_2(layer_3)
        x = self.upsample(x)
        layer_4 = self.activation(x)
        
        # Convolution №3
        x = self.conv2d_3(layer_4)
        layer_5 = x #self.sigmoid(x)
        
        return layer_5
    
    
    
class DenseEncoder(torch.nn.Module):
    """
    Dense encoder.
    
    Parameters
    ----------
    intput_dim : int
        Input dimension.
    latent_dim : int
        Latent representation dimension.
    """
    
    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Noise.
        self.dropout = torch.nn.Dropout(0.1)
        
        # Activation.
        self.activation = torch.nn.LeakyReLU()
        self.sigmoid = torch.nn.Sigmoid()
        
        # Dense.
        self.linear_1 = torch.nn.Linear(self.input_dim, 24)
        self.linear_2 = torch.nn.Linear(24, 16)
        self.linear_3 = torch.nn.Linear(16, self.latent_dim)
        
        
    def forward(self, x: torch.tensor) -> torch.tensor:
        # Dense №1
        x = self.dropout(x)
        x = self.linear_1(x)
        layer_1 = self.activation(x)
        
        # Dense №2
        x = self.dropout(layer_1)
        x = self.linear_2(x)
        layer_2 = self.activation(x)
        
        # Dense №3
        x = layer_2 #self.dropout(layer_2)
        x = self.linear_3(x)
        layer_3 = self.sigmoid(x)
        
        return layer_3
    
    
    
class DenseDecoder(torch.nn.Module):
    """
    Dense decoder.
    
    Parameters
    ----------
    latent_dim : int
        Latent representation dimension.
    output_dim : int
        Output dimension.
    """
        
    def __init__(self, latent_dim: int, output_dim: int):
        super().__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        
        # Activations.
        self.activation = torch.nn.LeakyReLU()
        
        # Dense.
        self.linear_1 = torch.nn.Linear(self.latent_dim, 16)
        self.linear_2 = torch.nn.Linear(16, 24)
        self.linear_3 = torch.nn.Linear(24, self.output_dim)
        
        
    def forward(self, x: torch.tensor) -> torch.tensor:
        # Dense №1
        x = self.linear_1(x)
        layer_1 = self.activation(x)
        
        # Dense №2
        x = self.linear_2(layer_1)
        layer_2 = self.activation(x)
        
        # Dense №3
        layer_3 = self.linear_3(layer_2)
        
        return layer_3



class Autoencoder(torch.nn.Module):
    """
    Autoencoder.
    
    Parameters
    ----------
    encoder : torch.nn.Module
        Encoder.
    decoder : torch.nn.Module
        Decoder.
    latent_dim : int
        Latent representation dimension.
    sigma : float
        Standard deviation of additive Gaussian noise,
        injected into the latent representation.
    """

    def __init__(self, encoder, decoder, sigma: float=0.1):
        super().__init__()
        #self.sigma = sigma
        
        # Encoder and decoder.
        self.encoder = encoder
        self.decoder = decoder
        
        # Noise.
        self.agn = AdditiveGaussianNoise(sigma=sigma, enabled_on_inference=False)
        
        
    def forward(self, x: torch.tensor) -> torch.tensor:
        latent = self.encoder(x)
        latent = self.agn(latent)
        
        return self.decoder(latent)
    
    
    def encode(self, x: torch.tensor) -> torch.tensor:
        return self.encoder(x)
    
    
    def decode(self, x: torch.tensor) -> torch.tensor:
        return self.decoder(x)



def evaluate_model(model, dataloader, loss, device) -> float:
    # Exit training mode.
    was_in_training = model.training
    model.eval()
  
    with torch.no_grad():
        avg_loss = 0.0
        total_samples = 0
        for batch in dataloader:
            x, y = batch
            batch_size = x.shape[0]
            
            y_pred = model(x.to(device))
            _loss = loss(y_pred, y.to(device))

            avg_loss += _loss.item() * batch_size
            total_samples += batch_size
            
        avg_loss /= total_samples
        
    # Return to the original mode.
    model.train(was_in_training)
    
    return avg_loss



def train_autoencoder(autoencoder, train_dataloader, test_dataloader, autoencoder_loss,
                      device, n_epochs: int=10, callback: callable=None) -> dict():
    autoencoder_opt = torch.optim.Adam(autoencoder.parameters(), lr=1e-2)
    
    autoencoder_metrics = {
        "train_loss" : [],
        "test_loss" : [],
    }
    
    for epoch in range(1, n_epochs + 1):
        print(f"Epoch №{epoch}")
        
        sum_loss = 0.0
        total_samples = 0
        for index, batch in tqdm(enumerate(train_dataloader)):
            x, y = batch
            batch_size = x.shape[0]
            
            autoencoder_opt.zero_grad()
            y_pred = autoencoder(x.to(device))
            _loss = autoencoder_loss(y_pred, y.to(device))
            _loss.backward()
            autoencoder_opt.step()
            
            sum_loss += _loss.item() * len(batch)
            total_samples += len(batch)
            
        autoencoder_metrics["train_loss"].append(sum_loss / total_samples)
        
        #train_loss = evaluate_model(autoencoder, train_dataloader, autoencoder_loss, device)
        #autoencoder_metrics["train_loss"].append(train_loss)
        test_loss = evaluate_model(autoencoder, test_dataloader, autoencoder_loss, device)
        autoencoder_metrics["test_loss"].append(test_loss)
        
        if not (callback is None):
            callback(autoencoder, autoencoder_metrics)
        
    return autoencoder_metrics