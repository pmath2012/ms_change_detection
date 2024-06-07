from __future__ import annotations
import cv2
import torch
import numpy as np
from typing import Type, List, Tuple
from torch import Tensor
from dataclasses import dataclass, field
from torch import nn

def tensor2cam(image, cam):
    image_with_heatmap = image2cam(
        image.permute(1, 2, 0).cpu().numpy(), cam.detach().cpu().numpy()
    )

    return torch.from_numpy(image_with_heatmap)


def image2cam(image, cam, eps=1e-6):
    h, w, c = image.shape

    cam -= np.min(cam)
    cam /= (np.max(cam)+eps)  # Normalize between 0-1
    cam = cv2.resize(cam, (h, w))
    cam = np.uint8(cam * 255.0)

    img_with_cam = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
    img_with_cam = cv2.cvtColor(img_with_cam, cv2.COLOR_BGR2RGB)
    img_with_cam = img_with_cam + (image * 255)
    img_with_cam /= np.max(img_with_cam)

    return img_with_cam

def find_last_layer(x: torch.Tensor, module: nn.Module, of_type: Type) -> nn.Module:
    """Utility function that return the last layer of a given type


    :Example:

    >>> x = torch.rand((1,3,224,224))
    >>> model = ResNet.resnet18()
    >>> find_last_layer(x, module, nn.Conv2d)

    Args:
        x (torch.Tensor): [description]
        module (nn.Module): [description]
        of_type (Type): [description]

    Returns:
        nn.Module: [description]
    """
    tr = Tracker(module)
    tr(x[0],x[1])

    layer = None
    # iterate backward so we save time!
    for m in tr.traced[::-1]:
        if isinstance(m, of_type):
            layer = m
            break
    assert layer != None, f"layer of type {of_type} not found in {module.__name__}"

    return layer

@dataclass
class Tracker:
    """This class tracks all the operations of a given module by performing a forward pass.

    Example:

        >>> import torch
        >>> import torch.nn as nn
        >>> from glasses.utils import Tracker
        >>> model = nn.Sequential(nn.Linear(2, 64), nn.ReLU(), nn.Linear(64,10), nn.ReLU())
        >>> tr = Tracker(model)
        >>> tr(x1, x2)
        >>> print(tr.traced) # all operations
        >>> print('-----')
        >>> print(tr.parametrized) # all operations with learnable params

        outputs

        ``[Linear(in_features=2, out_features=64, bias=True),
        ReLU(),
        Linear(in_features=64, out_features=10, bias=True),
        ReLU()]
        -----
        [Linear(in_features=2, out_features=64, bias=True),
        Linear(in_features=64, out_features=10, bias=True)]``
    """

    module: nn.Module
    traced: List[nn.Module] = field(default_factory=list)
    handles: list = field(default_factory=list)

    def _forward_hook(self, m, inputs: Tuple[Tensor, Tensor], outputs: Tensor):
        has_not_submodules = (
            len(list(m.modules())) == 1
            or isinstance(m, nn.Conv2d)
            or isinstance(m, nn.BatchNorm2d)
        )
        if has_not_submodules:
            self.traced.append(m)

    def __call__(self, x1: Tensor, x2: Tensor):
        for m in self.module.modules():
            self.handles.append(m.register_forward_hook(self._forward_hook))
        self.module(x1, x2)
        list(map(lambda x: x.remove(), self.handles))
        return self

    @property
    def parametrized(self):
        # check the len of the state_dict keys to see if we have learnable params
        return list(filter(lambda x: len(list(x.state_dict().keys())) > 0, self.traced))
