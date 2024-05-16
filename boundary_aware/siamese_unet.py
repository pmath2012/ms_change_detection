import torch
from torch import nn
from glasses.models import VisionModule
from glasses.models.segmentation.unet import UNetEncoder, UNetDecoder

from ..utils.siamese_heads import DifferenceNetwork, ConcatNetwork
class SiameseNetwork(VisionModule):
    def __init__(self,
                 in_channels=1,
                 out_channels=1,
                 encoder = UNetEncoder,
                 boundary_decoder = UNetDecoder,
                 mask_decoder = UNetDecoder,
                 head_nw = "concat",
                 **kwargs,):
        super().__init__()
        self.encoder = encoder(in_channels=in_channels, **kwargs)
        self.boundary_decoder = boundary_decoder(
            lateral_widths=self.encoder.features_widths[::-1],
            start_features=self.encoder.widths[-1],
            **kwargs,
        )
        self.mask_decoder = mask_decoder(
            lateral_widths=self.encoder.features_widths[::-1],
            start_features=self.encoder.widths[-1],
            **kwargs,
        )

        bmfb_in = self.boundary_decoder.widths[-1] + self.mask_decoder.widths[-1]
        bmfb_out = bmfb_in // 2
        self.bmfb_conv1 = nn.Conv2d(bmfb_in, bmfb_out, kernel_size=1)
        self.bmfb_conv2 = nn.Conv2d(bmfb_out, bmfb_out, kernel_size=1)

        if head_nw=="concat":
            self.mask_head = ConcatNetwork(in_features=bmfb_out*2) # *2 for concat features of mask1 and mask2
            self.boundary_head = ConcatNetwork(in_features=bmfb_in)
        elif head_nw=="diff":
            self.mask_head = DifferenceNetwork(in_features=bmfb_out) 
            self.boundary_head = DifferenceNetwork(in_features=bmfb_out)
        else:
            raise NotImplementedError("Head Network not specified")


    def forward_boundary(self, x):
        self.encoder.features
        x = self.encoder(x)
        # encoder must have a .features
        features = self.encoder.features
        self.boundary_residuals = features[::-1]
        self.boundary_residuals.extend([None] * (len(self.boundary_decoder.layers) - len(self.boundary_residuals)))

        x = self.boundary_decoder(x, self.boundary_residuals)

        return x

    def forward_mask(self, x):
        self.encoder.features
        x = self.encoder(x)
        # encoder must have a .features
        features = self.encoder.features
        self.mask_residuals = features[::-1]
        self.mask_residuals.extend([None] * (len(self.mask_decoder.layers) - len(self.mask_residuals)))

        x = self.mask_decoder(x, self.mask_residuals)

        return x

    def forward_once(self, x):
        x_b = self.forward_boundary(x)
        x_m = self.forward_mask(x)

        x_bmfb = torch.cat([x_b, x_m],1)
        x_bmfb = self.bmfb_conv1(x_bmfb)
        x_bmfb = self.bmfb_conv2(x_bmfb)

        return x_bmfb, x_b
    
    def forward(self, x1, x2):
        x1, b1 = self.forward_once(x1)
        x2, b2 = self.forward_once(x2)

        out_mask = self.mask_head(x1,x2)
        out_boundary = self.boundary_head(b1, b2)

        return out_mask, out_boundary
