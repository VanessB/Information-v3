import torch

class AutoencoderDataset(torch.utils.data.Dataset):
    """
    Construct dataset for autoencoder training from another dataset.
    """
    
    def __init__(self, dataset, dim: int=0):
        """
        Initialization.
        
        Parameters
        ----------
        dataset
            The dataset from which to make dataset for the autoencoder.
        dim : int, optional
            The number of the subelement (in each entry) to be repeated.
        """
        
        self.dataset = dataset
        self.dim = dim
        
        
    def __len__(self):
        return len(self.dataset)
    
    
    def __getitem__(self, index):
        x = self.dataset[index][self.dim]
        return (x, x)