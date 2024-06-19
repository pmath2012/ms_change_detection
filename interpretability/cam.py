import torch
import cv2
import numpy as np
from interpretability.utils import find_last_layer

class GradCAM:
    def __init__(self, model, model_name,target_layer=None, probe_encoder=False, with_boundary=False):
        self.model = model
        self.target_layer = target_layer
        self.probe_encoder = probe_encoder
        self.gradients = None
        self.activations = None
        self.with_boundary = with_boundary
        self.model_name = model_name
        
        if self.probe_encoder:
            self._register_hooks_all_layers()
        else:
            self._register_hooks(target_layer)

    def _register_hooks(self, layer):
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        def forward_hook(module, input, output):
            self.activations = output

        layer.register_forward_hook(forward_hook)
        layer.register_full_backward_hook(backward_hook)

    def _register_hooks_all_layers(self):
        def backward_hook(module, grad_in, grad_out):
            module.gradients = grad_out[0]

        def forward_hook(module, input, output):
            module.activations = output
        encoder = self._get_encoder()
        for module in encoder.layers:
            module.register_forward_hook(forward_hook)
            module.register_full_backward_hook(backward_hook)

    def _get_encoder(self):
        if self.model_name == "siamese_unet":
            return self.model.base_model.encoder
        if self.model_name == "pfpn":
            return self.model.model.encoder
        elif self.model_name == "boundary_aware":
            return self.model.encoder
        else:
            raise NotImplementedError("Model not supported")
        
    def _generate_cam(self, gradients, activations, input_image):
        # Get gradients and activations
        gradients = gradients.data.cpu().numpy()[0]
        activations = activations.data.cpu().numpy()[0]

        # Global average pooling of gradients
        weights = np.mean(gradients, axis=(1, 2))

        # Weighted sum of activations
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        # Apply ReLU
        cam = np.maximum(cam, 0)

        # Normalize the heatmap
        cam = cam - np.min(cam)
        cam = cam / (np.max(cam)+1e-10)
        cam = cv2.resize(cam, (input_image.shape[3], input_image.shape[2]))

        return cam

    def generate_cam(self, input_image1, input_image2):
        # Forward pass
        if self.with_boundary:
            output, _ = self.model(input_image1, input_image2)
        else:
            output = self.model(input_image1, input_image2)
        if isinstance(output, tuple):
            output = output[0]  # If Siamese Network returns a tuple (e.g., similarity score)

        # Zero the gradients
        self.model.zero_grad()

        # Sum the output to get a scalar value for backpropagation
        output_sum = output.sum()

        # Backward pass
        output_sum.backward(retain_graph=True)

        if self.probe_encoder:
            cams = []
            encoder = self._get_encoder()
            for name, module in encoder.named_modules():
                if hasattr(module, 'activations') and hasattr(module, 'gradients'):
                    cam1 = self._generate_cam(module.gradients, module.activations, input_image1)
                    cam2 = self._generate_cam(module.gradients, module.activations, input_image2)
                    cams.append((name, (cam1, cam2)))
            return cams, output
        else:
            cam1 = self._generate_cam(self.gradients, self.activations, input_image1)
            cam2 = self._generate_cam(self.gradients, self.activations, input_image2)
            return (cam1, cam2), output

def show_cam_on_image(img, mask):
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    heatmap = cv2.applyColorMap(np.uint8(255 * (1-mask)), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)