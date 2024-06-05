from __future__ import annotations
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
from glasses.interpretability import Interpretability
from typing import Callable
from .ba_storage import ForwardModuleStorage, BackwardModuleStorage
from glasses.interpretability.utils import tensor2cam
from typing import Type, List, Tuple
from torch import Tensor
from dataclasses import dataclass, field


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

    def __call__(self, x1: Tensor, x2: Tensor) -> Tracker:
        for m in self.module.modules():
            self.handles.append(m.register_forward_hook(self._forward_hook))
        self.module(x1, x2)
        list(map(lambda x: x.remove(), self.handles))
        return self

    @property
    def parametrized(self):
        # check the len of the state_dict keys to see if we have learnable params
        return list(filter(lambda x: len(list(x.state_dict().keys())) > 0, self.traced))


class GradCamResultSiam:
    def __init__(
        self,
        imgs,
        cams,
        postpreocessing: Callable[[torch.Tensor], torch.Tensor],
        output: torch.Tensor,
        boundary: torch.Tensor,
    ):
        self.img1 = imgs[0]
        self.img2 = imgs[1]
        self.cams = cams
        self.output = output
        self.boundary = boundary
        self.postpreocessing = postpreocessing
    
    def normalize_cam(self, cam):
        cam = cam - cam.min()
        cam = cam / cam.max()
        return cam

    def show(self, *args, **kwargs):
        img1 = self.img1
        if self.postpreocessing is not None:
            img1 = self.postpreocessing(self.img2)
        img2 = self.img2
        if self.postpreocessing is not None:
            img2 = self.postpreocessing(self.img2)

        outs = []
        for i, cam in enumerate(self.cams):
            cam_on_img1 = self.normalize_cam(tensor2cam(img1.squeeze(0), cam))
            cam_on_img2 = self.normalize_cam(tensor2cam(img2.squeeze(0), cam))
            outs.append((cam_on_img1, cam_on_img2))
        # output = torch.sigmoid(self.output)
        # boundary = torch.sigmoid(self.boundary
        return {"predicted": self.output,
                "boundary": self.boundary,
                "cams": outs}
            
            

class GradCamSiam(Interpretability):
    """
    Implementation of GradCam proposed in `Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization <https://arxiv.org/abs/1610.02391>`_
    """
    
    def __call__(
        self,
        x: torch.Tensor,
        module: nn.Module,
        layer: nn.Module = None,
        target: int = None,
        ctx: torch.Tensor = None,
        postprocessing: Callable[[torch.Tensor], torch.Tensor] = None,
        probe_encoder=False,
    ) -> GradCamResultSiam:
        """Run GradCam on the input given a model

        Args:
            x (torch.Tensor): Input tensor, e.g. an image
            module (nn.Module): Model
            layer (nn.Module, optional): The layer we wish to interpreter, if `None` then the last conv layer will be used. Defaults to None.
            target (int, optional): The target tensor, if `None` the model output (after softmax and argmax) wil be used. Defaults to None.
            ctx (torch.Tensor, optional): The tensor w.r we derive, if `None` we will use the one-hot encoding of the target. Defaults to None.
            postprocessing (Callable[[torch.Tensor], torch.Tensor], optional): A function used to post process the output, e.g. de-normalize. Defaults to None.

        Returns:
            GradCamResult: The result of the gradcam, you can call `.show` to see it.
        """
        cams = []
        if probe_encoder:
            layers = module.encoder.layers
        else:
            layers = [find_last_layer(x, module, nn.Conv2d) if layer is None else layer]
        for layer in layers:
            # register forward and backward storages
            features_storage = ForwardModuleStorage(module, [layer], debug=True)
            gradients_storage = BackwardModuleStorage([layer], debug=True)
            x1 = Variable(x[0], requires_grad=True)
            x2 = Variable(x[1], requires_grad=True)

            out, out_b = module(x1, x2)

            if target is None:
                target = torch.argmax(torch.softmax(out, dim=1))

            if ctx is None:
                ctx = torch.zeros(out.size()).to(x1.device)
                ctx[0][int(target)] = 1

            out.backward(gradient=ctx)
            # out_b.backward(gradient=ctx)

        
            # get back the weights and the gradients
            
            features = features_storage[layer]
            
            grads = gradients_storage[layer][0]
            
            # compute grad cam
            avg_channel_grad = F.adaptive_avg_pool2d(grads.data, 1)
            cam = F.relu(torch.sum(features * avg_channel_grad, dim=1)).squeeze(0)
            cams.append(cam)
        return GradCamResultSiam((x[0].detach(), x[1].detach()), cams, postprocessing, out.detach(), out_b.detach())
        
    
