import torch

from torch import nn

from glasses.models.segmentation.unet import UNetEncoder, UNetDecoder
from glasses.models.segmentation.base import VisionModule

class BAUNet(VisionModule):
    def __init__(self,
                 in_channels,
                 out_channels,
                 encoder = UNetEncoder,
                 boundary_decoder = UNetDecoder,
                 mask_decoder = UNetDecoder,
                 subtraction=False,
                 **kwargs):
        super().__init__()
        #super(BAUNet, self).__init__()
        self.subtraction = subtraction
        if subtraction:
            in_channels = in_channels+1
        self.encoder = encoder(in_channels=in_channels, **kwargs)
        self.boundary_decoder = boundary_decoder(
            lateral_widths=self.encoder.features_widths[::-1],
            start_features=self.encoder.widths[-1],
            **kwargs
        )
        self.mask_decoder = mask_decoder(
            lateral_widths=self.encoder.features_widths[::-1],
            start_features=self.encoder.widths[-1],
            **kwargs,
        )

        bmfb_in = self.boundary_decoder.widths[-1] + self.mask_decoder.widths[-1]
        bmfb_out = bmfb_in // 2
        self.boundary_head = nn.Conv2d(self.boundary_decoder.widths[-1], out_channels, kernel_size=1)

        self.bmfb_conv1 = nn.Conv2d(bmfb_in, bmfb_out, kernel_size=1)
        self.bmfb_conv2 = nn.Conv2d(bmfb_out, bmfb_out, kernel_size=1)

        self.mask_head = nn.Conv2d(bmfb_out, out_channels, kernel_size=1)


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

    def forward(self, x1, x2):
        if self.subtraction:
            sub = torch.abs(torch.subtract(x2, x1))
            x = torch.cat([x1, x2, sub], dim=1)
        else:
            x = torch.cat([x1,x2], 1)
        x_b = self.forward_boundary(x)
        x_m = self.forward_mask(x)

        x_bmfb = torch.cat([x_b, x_m],1)
        x_bmfb = self.bmfb_conv1(x_bmfb)
        x_bmfb = self.bmfb_conv2(x_bmfb)

        out_mask = self.mask_head(x_bmfb)
        out_boundary = self.boundary_head(x_b)

        return out_mask, out_boundary