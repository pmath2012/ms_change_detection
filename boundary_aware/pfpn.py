from torch import nn
import torch
from glasses.models.classification.resnet import ResNetEncoder
from glasses.models.segmentation.fpn import FPNSegmentationBranch, FPNDecoder, PFPNSegmentationLayer, PFPN, Merge, FPN
from glasses.models.segmentation.fpn import PFPNDecoder
from glasses.models.base import VisionModule
from functools import partial
from glasses.models import AutoModel

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_channels, in_channels // ratio, kernel_size=1, stride=1, padding=0)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels // ratio, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return torch.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=(kernel_size-1)//2)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return torch.sigmoid(x)

class CBAMBlock(nn.Module):
    def __init__(self, in_channels, ratio=16, kernel_size=7):
        super(CBAMBlock, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        channel_out = self.channel_attention(x) * x
        spatial_out = self.spatial_attention(x) * x
        return channel_out + spatial_out

AttentionPFPNSegmentationBranch = partial(FPNSegmentationBranch, block=CBAMBlock, layer=PFPNSegmentationLayer)
AttentionPFPNDecoder = partial(FPNDecoder, segmentation_branch=AttentionPFPNSegmentationBranch)

class AttentionPFPN(FPN):
    def __init__(self, *args, n_classes: int = 2, decoder: nn.Module = AttentionPFPNDecoder, **kwargs):
        super().__init__(*args, decoder=decoder, **kwargs)
        self.head = nn.Sequential(
            Merge(),
            nn.UpsamplingNearest2d(scale_factor=4),
            nn.Conv2d(self.decoder.widths[-1], n_classes, kernel_size=1),
        )

class PFPNSeg(nn.Module):
    def __init__(self, in_channels, out_channels, attention=False, backbone="resnet50"):
        super().__init__()
        if attention:
            self.model = AttentionPFPN.from_encoder(model=partial(AutoModel.from_name, backbone),
                    in_channels = in_channels, n_classes=out_channels)
        else:
            self.model = PFPN.from_encoder(model=partial(AutoModel.from_name, backbone),
                    in_channels = in_channels, n_classes=out_channels)
    def forward(self, x1, x2):
        x =torch.cat([x1, x2], 1)
        return self.model(x)

class MultiTaskPFPN(VisionModule):
    def __init__(self, 
                 in_channels,
                 out_channels,
                 encoder = ResNetEncoder,
                 boundary_decoder = PFPNDecoder,
                 mask_decoder = PFPNDecoder,
                 **kwargs,):
        super().__init__()
        self.encoder = encoder(in_channels=in_channels, **kwargs)
        self.mask_decoder = mask_decoder(
            lateral_widths=self.encoder.features_widths[::-1],
            start_features=self.encoder.widths[-1],
            **kwargs,
        )
        self.boundary_decoder = boundary_decoder(
            lateral_widths=self.encoder.features_widths[::-1],
            start_features=self.encoder.widths[-1],
            **kwargs,
        )

        self.boundary_process = nn.Sequential(Merge(), nn.UpsamplingNearest2d(scale_factor=4))
        self.mask_process = nn.Sequential(Merge(), nn.UpsamplingNearest2d(scale_factor=4))

        head_in_features = self.mask_decoder.widths[-1] + self.boundary_decoder.widths[-1]
        self.boundary_head = nn.Conv2d(self.boundary_decoder.widths[-1], out_channels=out_channels, kernel_size=1)
        self.mask_head = nn.Conv2d(head_in_features, out_channels=out_channels, kernel_size=1)

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
        x = torch.cat([x1, x2], 1)
        x_b = self.forward_boundary(x)
        x_m = self.forward_mask(x)
        
        # pfpn upsampling step
        x_b = self.boundary_process(x_b)
        x_m = self.mask_process(x_m)

        # fusing mask and boundary features
        x_bmf = torch.cat([x_m, x_b],1)

        out_mask = self.mask_head(x_bmf)
        out_boundary = self.boundary_head(x_b)
        
        return out_mask, out_boundary