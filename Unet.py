# modify the MONAI Unet that split decoder and encoder 

from typing import Sequence, Union

import torch
import torch.nn as nn

from monai.networks.blocks.convolutions import Convolution, ResidualUnit
from monai.networks.layers.factories import Act, Norm
from monai.networks.layers.simplelayers import SkipConnection
from monai.utils import alias, export

class UNet(nn.Module):
    def __init__(
        self,
        dimensions: int,
        in_channels: int,
        out_channels: int,
        channels: Sequence[int],
        strides: Sequence[int],
        kernel_size: Union[Sequence[int], int] = 3,
        up_kernel_size: Union[Sequence[int], int] = 3,
        num_res_units: int = 0,
        act=Act.PRELU,
        norm=Norm.INSTANCE,
        dropout=0.0,
    ) -> None:
        """
        Enhanced version of UNet which has residual units implemented with the ResidualUnit class.
        The residual part uses a convolution to change the input dimensions to match the output dimensions
        if this is necessary but will use nn.Identity if not.
        Refer to: https://link.springer.com/chapter/10.1007/978-3-030-12029-0_40.

        Args:
            dimensions: number of spatial dimensions.
            in_channels: number of input channels.
            out_channels: number of output channels.
            channels: sequence of channels. Top block first.
            strides: convolution stride.
            kernel_size: convolution kernel size. Defaults to 3.
            up_kernel_size: upsampling convolution kernel size. Defaults to 3.
            num_res_units: number of residual units. Defaults to 0.
            act: activation type and arguments. Defaults to PReLU.
            norm: feature normalization type and arguments. Defaults to instance norm.
            dropout: dropout ratio. Defaults to no dropout.
        """
        super().__init__()

        self.dimensions = dimensions
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = channels
        self.strides = strides
        self.kernel_size = kernel_size
        self.up_kernel_size = up_kernel_size
        self.num_res_units = num_res_units
        self.act = act
        self.norm = norm
        self.dropout = dropout
        
        self.down1 = self.get_down_layer(in_channels, channels[0], strides[0], True)
        self.down2 = self.get_down_layer(channels[0], channels[1], strides[1], False)
        self.down3 = self.get_down_layer(channels[1], channels[2], strides[2], False)
        self.down4 = self.get_down_layer(channels[2], channels[3], strides[3], False)
        # 將最後一層與前一層用SkipConnection相加
        self.down5 = SkipConnection(self.get_bottom_layer(channels[3], channels[4]))
        
        self.up1 = self.get_up_layer(channels[3]+channels[4], channels[2], strides[3], False)
        self.up2 = self.get_up_layer(channels[3], channels[1], strides[2], False)
        self.up3 = self.get_up_layer(channels[2], channels[0], strides[1], False)
        self.up4 = self.get_up_layer(channels[1], out_channels, strides[0], True)
        
        '''
        #遞迴 不斷create_block 最終只有一個model 所以要解開遞迴 將每一層拉出來
        def create_block(
            inc: int, outc: int, channels: Sequence[int], strides: Sequence[int], is_top: bool
        ) -> nn.Sequential:
            """
            Builds the UNet structure from the bottom up by recursing down to the bottom block, then creating sequential
            blocks containing the downsample path, a skip connection around the previous block, and the upsample path.

            Args:
                inc: number of input channels.
                outc: number of output channels.
                channels: sequence of channels. Top block first.
                strides: convolution stride.
                is_top: True if this is the top block.
            """
            c = channels[0]
            #c = channels
            s = strides[0]
            #s = strides

            subblock: nn.Module
        
            if len(channels) > 2:
                subblock = create_block(c, c, channels[1:], strides[1:], False)  # continue recursion down
                #subblock = create_block(c, c, channels, strides, False)  # continue recursion down
                #subblock = self.get_bottom_layer(c, channels)
                upc = c * 2
            else:
                # the next layer is the bottom so stop recursion, create the bottom layer as the sublock for this layer
                subblock = self.get_bottom_layer(c, channels[1])
                #subblock = self.get_bottom_layer(c, channels)
                upc = c + channels[1]
                #upc = c + channels

            down = self.get_down_layer(inc, c, s, is_top)  # create layer in downsampling path
            up = self.get_up_layer(upc, outc, s, is_top)  # create layer in upsampling path
            return nn.Sequential(down, SkipConnection(subblock), up)
        
        #self.model = create_block(in_channels, out_channels, self.channels, self.strides, True)
        '''
    def get_down_layer(self, in_channels: int, out_channels: int, strides: int, is_top: bool) -> nn.Module:
        """
        Args:
            in_channels: number of input channels.
            out_channels: number of output channels.
            strides: convolution stride.
            is_top: True if this is the top block.
        """
        if self.num_res_units > 0:
            return ResidualUnit(
                self.dimensions,
                in_channels,
                out_channels,
                strides=strides,
                kernel_size=self.kernel_size,
                subunits=self.num_res_units,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
            )
        return Convolution(
            self.dimensions,
            in_channels,
            out_channels,
            strides=strides,
            kernel_size=self.kernel_size,
            act=self.act,
            norm=self.norm,
            dropout=self.dropout,
        )

    def get_bottom_layer(self, in_channels: int, out_channels: int) -> nn.Module:
        """
        Args:
            in_channels: number of input channels.
            out_channels: number of output channels.
        """
        return self.get_down_layer(in_channels, out_channels, 1, False)

    def get_up_layer(self, in_channels: int, out_channels: int, strides: int, is_top: bool) -> nn.Module:
        """
        Args:
            in_channels: number of input channels.
            out_channels: number of output channels.
            strides: convolution stride.
            is_top: True if this is the top block.
        """
        conv: Union[Convolution, nn.Sequential]

        conv = Convolution(
            self.dimensions,
            in_channels,
            out_channels,
            strides=strides,
            kernel_size=self.up_kernel_size,
            act=self.act,
            norm=self.norm,
            dropout=self.dropout,
            conv_only=is_top and self.num_res_units == 0,
            is_transposed=True,
        )

        if self.num_res_units > 0:
            ru = ResidualUnit(
                self.dimensions,
                out_channels,
                out_channels,
                strides=1,
                kernel_size=self.kernel_size,
                subunits=1,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
                last_conv_only=is_top,
            )
            conv = nn.Sequential(conv, ru)

        return conv
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ''' Divide the create_block to encoder & decoder '''
            
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        dlast = self.down5(d4)
        #dlast = torch.cat([d4, d5], dim=1)
        skip1 = self.up1(dlast) 
        u1 = torch.cat([skip1, d3], dim=1)
        skip2 = self.up2(u1)
        u2 = torch.cat([skip2, d2], dim=1)
        skip3 = self.up3(u2)
        u3 = torch.cat([skip3, d1], dim=1)
        ori = self.up4(u3)
        
        #x = self.model(x)
        #x = self.
        return ori

Unet = unet = UNet
