import torch

class AdditiveGaussianNoise(torch.nn.Module):
    """
    Additive Gaussian noise.
    """

    def __init__(self, sigma : float=0.1, relative_scale : bool=True, enabled_on_inference : bool=False):
        """
        Initialization.

        Parameters
        ----------
        sigma : float, optional
            Standard deviation of additive Gaussian noise.
        relative_scale : bool, optional
            Scale the noise to preserve noise-to-signal ratio.
        enabled_on_inference : bool, optional
            Inject noise on inference.
        """
        
        super().__init__()
        self.sigma = sigma
        self.relative_scale = relative_scale
        self.enabled_on_inference = enabled_on_inference
        
        # A separate parameter for noise.
        # Needed in order to generate noise on the desired device.
        #self.noise = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        #self.noise.requires_grad_(False)
        self.register_buffer('noise', torch.tensor(0.0, dtype=torch.float32))  # Is preferable in case of multiple devices.
        

    def forward(self, x):
        if (self.training or self.enabled_on_inference) and self.sigma != 0:
            scale = self.sigma * x.detach() if self.relative_scale else self.sigma
            sampled_noise = self.noise.repeat(*x.size()).normal_() * scale
            x = x + sampled_noise
        return x