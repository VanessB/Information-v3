import torch

class AdditiveGaussianNoise(torch.nn.Module):
    """
    Аддитивный гауссов шум.

    Параметры
    ---------
    sigma : float, optional
        Стандартное отклонение аддитивного гауссова шума.
    relative_scale : bool, optional
        Скалировать ли шум входными значениями?
    enabled_on_inference : bool, optional
        Добавляется ли шум вне обучения.
    """

    def __init__(self, sigma : float=0.1, relative_scale : bool=True, enabled_on_inference : bool=False):
        """
        Конструктор класса.

        Параметры
        ---------
        sigma : float, optional
            Стандартное отклонение аддитивного гауссова шума.
        relative_scale : bool, optional
            Скалировать ли шум входными значениями?
        enabled_on_inference : bool, optional
            Добавляется ли шум вне обучения.
        """
        
        super().__init__()
        self.sigma = sigma
        self.relative_scale = relative_scale
        self.enabled_on_inference = enabled_on_inference
        
        # Отдельный параметр для шума.
        # Нужен для того, чтобы генерировать шум сразу на нужном устройстве.
        #self.noise = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        #self.noise.requires_grad_(False)
        self.register_buffer('noise', torch.tensor(0.0, dtype=torch.float32))  # Предпочтительней в случае нескольких устройств.
        

    def forward(self, x):
        if (self.training or self.enabled_on_inference) and self.sigma != 0:
            scale = self.sigma * x.detach() if self.relative_scale else self.sigma
            sampled_noise = self.noise.repeat(*x.size()).normal_() * scale
            x = x + sampled_noise
        return x