#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Simon Schaefer
# Description : Task aware image downscaling autoencoder model - SCALING.
#               Direct scaling factor 4.
# =============================================================================
import torch
from torch import nn


def build_net():
    return AETAD_4Direct()

class _Resblock_(nn.Module):
    """ Residual convolutional block consisting of two convolutional
    layers, a RELU activation in between and a residual connection from
    start to end. The inputs size (=s) is therefore contained. The number
    of channels is contained as well, but can be adapted (=c). """

    __constants__ = ['channels']

    def __init__(self, c):
        super(_Resblock_, self).__init__()
        self.filter_block = nn.Sequential(
            nn.Conv2d(c, c, 3, stride=1, padding=1, bias=True),     # b, c, s, s
            nn.ReLU(True),                                          # b, c, s, s
            nn.Conv2d(c, c, 3, stride=1, padding=1, bias=True)      # b, c, s, s
        )
        self.channels = c

    def forward(self, x):
        return x + self.filter_block(x)

    def extra_repr(self):
        return 'channels={}'.format(self.channels)

class _ReversePixelShuffle_(nn.Module):
    """ Reverse pixel shuffeling module, i.e. rearranges elements in a tensor
    of shape (*, C, H*r, W*r) to (*, C*r^2, H, W). Inverse implementation according
    to https://pytorch.org/docs/0.3.1/_modules/torch/nn/functional.html#pixel_shuffle. """

    __constants__ = ['downscale_factor']

    def __init__(self, downscale_factor):
        super(_ReversePixelShuffle_, self).__init__()
        self.downscale_factor = downscale_factor

    def forward(self, input):
        _, c, h, w = input.shape
        assert all([x % self.downscale_factor == 0 for x in [h, w]])
        return self.inv_pixel_shuffle(input, self.downscale_factor)

    def extra_repr(self):
        return 'downscale_factor={}'.format(self.downscale_factor)

    @staticmethod
    def inv_pixel_shuffle(input, downscale_factor):
        batch_size, in_channels, height, width = input.size()
        out_channels = in_channels * (downscale_factor ** 2)
        height //= downscale_factor
        width //= downscale_factor
        # Reshape input to new shape.
        input_view = input.contiguous().view(
            batch_size, in_channels, height, downscale_factor,
            width, downscale_factor)
        shuffle_out = input_view.permute(0,1,3,5,2,4).contiguous()
        return shuffle_out.view(batch_size, out_channels, height, width)

class AETAD_4Direct(nn.Module):

    def __init__(self):
        super(AETAD_4Direct, self).__init__()
        # Build encoding part.
        self._downscaling = nn.Sequential(
            nn.Conv2d(3, 4, 3, stride=1, padding=1),
            nn.Conv2d(4, 4, 3, stride=1, padding=1),
            _ReversePixelShuffle_(downscale_factor=4),
        )
        self._res_en1 = _Resblock_(64)
        self._res_en2 = _Resblock_(64)
        self._res_en3 = _Resblock_(64)
        self._conv_en1 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self._conv_en2 = nn.Conv2d(64, 3, 3, stride=1, padding=1)
        # Build decoding part.
        self._conv_de1 = nn.Conv2d(3, 64, 3, stride=1, padding=1)
        self._res_de1 = _Resblock_(64)
        self._res_de2 = _Resblock_(64)
        self._res_de3 = _Resblock_(64)
        self._conv_de2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self._upscaling = nn.Sequential(
            nn.Conv2d(64, 256, 3, stride=1, padding=1),
            nn.PixelShuffle(upscale_factor=4),
            nn.Conv2d(16, 3, 3, stride=1, padding=1)
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:              # b, 3, p, p
        x = self._downscaling(x)                                    # b, 64, p/2, p/2
        residual = x
        x = self._res_en1.forward(x)                                # b, 64, p/2, p/2
        x = self._res_en2.forward(x)                                # b, 64, p/2, p/2
        x = self._res_en3.forward(x)                                # b, 64, p/2, p/2
        x = self._conv_en1(x)                                       # b, 64, p/2, p/2
        x = torch.add(residual, x)                                  # b, 64, p/2, p/2
        x = self._conv_en2(x)                                       # b, 3, p/2, p/2
        return x

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        x = self._conv_de1(x)                                       # b, 64, p/2, p/2
        residual = x
        x = self._res_de1.forward(x)                                # b, 64, p/2, p/2
        x = self._res_de2.forward(x)                                # b, 64, p/2, p/2
        x = self._res_de3.forward(x)                                # b, 64, p/2, p/2
        x = self._conv_de2(x)                                       # b, 64, p/2, p/2
        x = torch.add(residual, x)                                  # b, 64, p/2, p/2
        x = self._upscaling(x)                                      # b, 3, p, p
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))