import torch

class AutoencoderDataset(torch.utils.data.Dataset):
    """
    Набор данных для автокодировщика из другого набора данных.
    
    Параметры
    ---------
    dataset
        Набор данных, из которого требуется сделать данные для автокодировщика.
    dim : int
        Номер подэелемента, который требуется повторить.
    """
    
    def __init__(self, dataset, dim: int=0):
        """
        Конструктор класса.
        
        Параметры
        ---------
        dataset
            Набор данных, из которого требуется сделать данные для автокодировщика.
        dim : int, optional
            Номер подэелемента, который требуется повторить.
        """
        
        self.dataset = dataset
        self.dim = dim
        
        
    def __len__(self):
        return len(self.dataset)
    
    
    def __getitem__(self, index):
        x = self.dataset[index][self.dim]
        return (x, x)