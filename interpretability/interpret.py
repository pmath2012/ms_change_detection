import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
from glasses.interpretability import Interpretability
from typing import Callable
from interpretability.storage import ForwardModuleStorage, BackwardModuleStorage
from glasses.interpretability.utils import tensor2cam
from interpretability.utils import find_last_layer

class GradCamResultSiam:
    def __init__(
        self,
        imgs,
        cams,
        postpreprocessing: Callable[[torch.Tensor], torch.Tensor],
        output: torch.Tensor,
        boundary: torch.Tensor,
    ):
        self.img1 = imgs[0]
        self.img2 = imgs[1]
        self.cams = cams
        self.output = output
        self.boundary = boundary
        self.postpreprocessing = postpreprocessing

    def show(self, *args, **kwargs):
        img1 = self.img1
        if self.postpreprocessing is not None:
            img1 = self.postpreprocessing(self.img2)
        img2 = self.img2
        if self.postpreprocessing is not None:
            img2 = self.postpreprocessing(self.img2)

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
        with_boundary=False,
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

            if with_boundary:
                out, out_b = module(x1, x2)
            else:
                out = module(x1,x2)

            if target is None:
                target = torch.argmax(torch.softmax(out, dim=1))

            if ctx is None:
                ctx = torch.zeros(out.size()).to(x1.device)
                ctx[0][int(target)] = 1

            out.backward(gradient=ctx)
            # get back the weights and the gradients
            
            features = features_storage[layer]
            
            grads = gradients_storage[layer][0]
            
            # compute grad cam
            avg_channel_grad = F.adaptive_avg_pool2d(grads.data, 1)
            cam = F.relu(torch.sum(features * avg_channel_grad, dim=1)).squeeze(0)
            cams.append(cam)
        
        if with_boundary:
            return GradCamResultSiam((x1.detach(), x2.detach()), cams, postprocessing, out.detach(), out_b.detach())
        else:
            return GradCamResultSiam((x1.detach(), x2.detach()), cams, postprocessing, out.detach(), None)
    
