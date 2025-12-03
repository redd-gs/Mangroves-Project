import numpy as np
import torch
from torchvision.transforms import functional as F
import torchvision.transforms as T
from typing import Tuple, Dict


class ParametricTransform(torch.nn.Module):
    def __init__(self):
        super().__init__()


class ExampleParametricTransform(ParametricTransform): 
    """
    Example of a parametric transform.
    """
    def forward(self, img: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        return img, {'example': {}}
    

class Compose(T.Compose):
    """
    Extension of torchvision.transforms.Compose to support parametric transforms.
    """
    def __call__(self, img: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        compose_parameters = {}
        for t in self.transforms:
            if isinstance(t, ParametricTransform):
                img, parameters = t(img)
                compose_parameters.update(parameters)
            else:
                img = t(img)
        return img, compose_parameters
    