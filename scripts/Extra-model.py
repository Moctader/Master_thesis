
                                        # UNET++ for better resoulation  against ground trouth


import torch
import torch.nn as nn
import torch.nn.functional as F
class conv_block_nested(nn.Module):

    def __init__(self, in_ch, mid_ch, out_ch):
        super(conv_block_nested, self).__init__()
        self.activation = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.bn2(x)
        output = self.activation(x)

        return output

class Nested_UNet(nn.Module):

    def __init__(self, in_ch=3, out_ch=3):
        super(Nested_UNet, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4]

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = conv_block_nested(in_ch, filters[0], filters[0])
        self.conv1_0 = conv_block_nested(filters[0], filters[1], filters[1])


        self.conv0_1 = conv_block_nested(filters[0] + filters[1], filters[0], filters[0])
        self.conv1_1 = conv_block_nested(filters[1] + filters[2], filters[1], filters[1])

        self.conv0_2 = conv_block_nested(filters[0]*2 + filters[1], filters[0], filters[0])
        self.conv1_2 = conv_block_nested(filters[1]*2 + filters[2], filters[1], filters[1])


        self.conv0_3 = conv_block_nested(filters[0]*3 + filters[1], filters[0], filters[0])
        self.conv1_3 = conv_block_nested(filters[1]*3 + filters[2], filters[1], filters[1])

        self.final = nn.Conv2d(filters[0], out_ch, kernel_size=1)

    def forward(self, x, out_shape=None):

        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.Up(x1_0)], 1))



        output = self.final(x0_1)
        if out_shape is not None:
            return F.interpolate(output, size=out_shape, mode='bilinear', align_corners=True)
        else:
            return output.permute(0, 2, 3, 1)





                                        # UNET
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        def double_conv(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.ReLU(inplace=True)
            )

        class UNet(nn.Module):

            def __init__(self, n_class=10):
                super().__init__()

                self.dconv_down1 = double_conv(3, 64)
                self.dconv_down2 = double_conv(64, 128)
                self.dconv_down3 = double_conv(128, 256)
                self.dconv_down4 = double_conv(256, 512)
                self.dconv_down5 = double_conv(512, 1024)

                self.maxpool = nn.MaxPool2d(2)
                self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

                self.dconv_up4 = double_conv(512 + 1024, 512)
                self.dconv_up3 = double_conv(256 + 512, 256)
                self.dconv_up2 = double_conv(128 + 256, 128)
                self.dconv_up1 = double_conv(128 + 64, 64)

                self.conv_last = nn.Conv2d(64, n_class, 1)

            def forward(self, x):
                conv1 = self.dconv_down1(x)
                x = self.maxpool(conv1)

                conv2 = self.dconv_down2(x)
                x = self.maxpool(conv2)

                conv3 = self.dconv_down3(x)
                x = self.maxpool(conv3)

                conv4 = self.dconv_down4(x)
                x = self.maxpool(conv4)

                x = self.dconv_down5(x)

                x = self.upsample(x)
                x = torch.cat([x, conv3], dim=1)

                x = self.dconv_up4(x)
                x = self.upsample(x)
                x = torch.cat([x, conv3], dim=1)

                x = self.dconv_up3(x)
                x = self.upsample(x)
                x = torch.cat([x, conv2], dim=1)

                x = self.dconv_up2(x)
                x = self.upsample(x)
                x = torch.cat([x, conv1], dim=1)

                x = self.dconv_up1(x)

                out = self.conv_last(x)

                return out




              ############3UNET

                class down(nn.Module):
                    def __init__(self, in_ch, out_ch):
                        super(down, self).__init__()
                        self.mpconv = nn.Sequential(
                            nn.MaxPool2d(2),
                            double_conv(in_ch, out_ch)
                        )

                    def forward(self, x):
                        x = self.mpconv(x)
                        return x

                class up(nn.Module):
                    def __init__(self, in_ch, out_ch, bilinear=True):
                        super(up, self).__init__()
                        self.bilinear = bilinear

                        self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)

                        self.conv = double_conv(in_ch, out_ch)

                    def forward(self, x1, x2):
                        if self.bilinear:
                            x1 = F.interpolate(x1, scale_factor=2, mode='bilinear', align_corners=True)
                        else:
                            x1 = self.up(x1)

                        # input is CHW
                        diffY = x2.size()[2] - x1.size()[2]
                        diffX = x2.size()[3] - x1.size()[3]

                        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                                        diffY // 2, diffY - diffY // 2])

                        # for padding issues, see
                        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
                        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

                        x = torch.cat([x2, x1], dim=1)
                        x = self.conv(x)
                        return x

                class outconv(nn.Module):
                    def __init__(self, in_ch, out_ch):
                        super(outconv, self).__init__()
                        self.conv = nn.Conv2d(in_ch, out_ch, 1)

                    def forward(self, x):
                        x = self.conv(x)
                        return x

                class double_conv(nn.Module):
                    '''(conv => BN => ReLU) * 2'''

                    def __init__(self, in_ch, out_ch):
                        super(double_conv, self).__init__()
                        self.conv = nn.Sequential(
                            nn.Conv2d(in_ch, out_ch, 3, padding=1),
                            nn.BatchNorm2d(out_ch),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(out_ch, out_ch, 3, padding=1),
                            nn.BatchNorm2d(out_ch),
                            nn.ReLU(inplace=True)
                        )

                    def forward(self, x):
                        x = self.conv(x)
                        return x

                class inconv(nn.Module):
                    def __init__(self, in_ch, out_ch):
                        super(inconv, self).__init__()
                        self.conv = double_conv(in_ch, out_ch)

                    def forward(self, x):
                        x = self.conv(x)
                        return x

                class UNet(nn.Module):
                    def __init__(self, classes=12):
                        super(UNet, self).__init__()
                        self.inc = inconv(3, 64)
                        self.down1 = down(64, 128)
                        self.down2 = down(128, 256)
                        self.down3 = down(256, 512)
                        self.down4 = down(512, 1024)

                        self.up1 = up(512 + 1024, 512)
                        self.up2 = up(256 + 512, 256)
                        self.up3 = up(128 + 256, 128)
                        self.up4 = up(128 + 64, 3)

                        self.outc = outconv(3, classes)

                    def forward(self, x):
                        x1 = self.inc(x)
                        x2 = self.down1(x1)
                        x3 = self.down2(x2)
                        x4 = self.down3(x3)
                        x5 = self.down4(x4)
                        x = self.up1(x5, x4)
                        x = self.up2(x, x3)
                        x = self.up3(x, x2)
                        x = self.up4(x, x1)
                        x = self.outc(x)
                        # return F.sigmoid(x)

                        return x

                    ###44444

import torch
import torch.nn as nn
import torch.nn.functional as F



class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self,
                 n_channels=1,
                 n_classes=10,
                 bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x, out_shape=None):
        #if out_shape is not None:
         #   return x.permute(0, 3, 1, 2)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits



    #### Working model unet
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class inconv(nn.Module):
        def __init__(self, in_ch, out_ch):
            super(inconv, self).__init__()
            self.conv = double_conv(in_ch, out_ch)

        def forward(self, x):
            x = self.conv(x)
            return x

    class double_conv(nn.Module):
        '''(conv => BN => ReLU) * 2'''

        def __init__(self, in_ch, out_ch):
            super(double_conv, self).__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )

        def forward(self, x):
            x = self.conv(x)
            return x

    class down(nn.Module):
        def __init__(self, in_ch, out_ch):
            super(down, self).__init__()
            self.mpconv = nn.Sequential(
                nn.MaxPool2d(2),
                double_conv(in_ch, out_ch)
            )

        def forward(self, x):
            x = self.mpconv(x)
            return x

    # Upsampling/ Unpooling/ Deconvolution/ Transpose Convolution, followed by two conv layers
    class up(nn.Module):
        def __init__(self, in_ch, out_ch, bilinear=True):
            super(up, self).__init__()

            #  would be a nice idea if the upsampling could be learned too,
            #  but my machine do not have enough memory to handle all those weights
            if bilinear:
                self.up = nn.Upsample(
                    scale_factor=2, mode='bilinear', align_corners=True)
            else:
                self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)

            self.conv = double_conv(in_ch, out_ch)

        def forward(self, x1, x2):
            x1 = self.up(x1)

            # input is CHW
            diffY = x2.size()[2] - x1.size()[2]
            diffX = x2.size()[3] - x1.size()[3]

            x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2))

            # for padding issues, see
            # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
            # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

            x = torch.cat([x2, x1], dim=1)
            x = self.conv(x)
            return x

    class outconv(nn.Module):
        def __init__(self, in_ch, out_ch):
            super(outconv, self).__init__()
            self.conv = nn.Conv2d(in_ch, out_ch, 1)

        def forward(self, x):
            x = self.conv(x)
            return x

    class UNet(nn.Module):
        def __init__(self, n_channels=3, n_classes=3):
            super(UNet, self).__init__()
            self.inc = inconv(n_channels, 64)  # First step of Contracting
            self.down1 = down(64, 128)  # Second step of Contracting
            self.down2 = down(128, 256)  # Third step of Contracting
            self.down3 = down(256, 512)  # Fourth step of Contracting
            self.down4 = down(512, 512)  # Bottleneck of U-Net
            self.up1 = up(1024, 256)  # First step of Expanding
            self.up2 = up(512, 128)  # Second step of Expanding
            self.up3 = up(256, 64)  # Third step of Expanding
            self.up4 = up(128, 64)  # Fourth step of Expanding
            # Output Conv layer with 1*1 filter
            self.outc = outconv(64, n_classes)

        def forward(self, x, out_shape=None):
            x1 = self.inc(x)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            x5 = self.down4(x4)
            x = self.up1(x5, x4)
            x = self.up2(x, x3)
            x = self.up3(x2, x)
            x = self.up4(x, x1)
            x = self.outc(x)
            # return F.sigmoid(x)
            if out_shape is not None:
                return F.interpolate(output, size=out_shape, mode='bicubic', align_corners=True)
            else:
                return x.permute(0, 2, 3, 1)

            import torch
            import torch.nn as nn
            import torch.nn.functional as F

            class Conv3x3GNReLU(nn.Module):
                def __init__(self, in_channels, out_channels, upsample=False):
                    super().__init__()
                    self.upsample = upsample
                    self.block = nn.Sequential(
                        nn.Conv2d(
                            in_channels, out_channels, (3, 3), stride=1, padding=1, bias=False
                        ),
                        nn.GroupNorm(32, out_channels),
                        nn.ReLU(inplace=True),
                    )

                def forward(self, x):
                    x = self.block(x)
                    if self.upsample:
                        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
                    return x

            class FPNBlock(nn.Module):
                def __init__(self, pyramid_channels, skip_channels):
                    super().__init__()
                    self.skip_conv = nn.Conv2d(skip_channels, pyramid_channels, kernel_size=1)

                def forward(self, x, skip=None):
                    x = F.interpolate(x, scale_factor=2, mode="nearest")
                    skip = self.skip_conv(skip)
                    x = x + skip
                    return x

            class SegmentationBlock(nn.Module):
                def __init__(self, in_channels, out_channels, n_upsamples=0):
                    super().__init__()

                    blocks = [Conv3x3GNReLU(in_channels, out_channels, upsample=bool(n_upsamples))]

                    if n_upsamples > 1:
                        for _ in range(1, n_upsamples):
                            blocks.append(Conv3x3GNReLU(out_channels, out_channels, upsample=True))

                    self.block = nn.Sequential(*blocks)

                def forward(self, x):
                    return self.block(x)

            class MergeBlock(nn.Module):
                def __init__(self, policy):
                    super().__init__()
                    if policy not in ["add", "cat"]:
                        raise ValueError(
                            "`merge_policy` must be one of: ['add', 'cat'], got {}".format(
                                policy
                            )
                        )
                    self.policy = policy

                def forward(self, x):
                    if self.policy == 'add':
                        return sum(x)
                    elif self.policy == 'cat':
                        return torch.cat(x, dim=1)
                    else:
                        raise ValueError(
                            "`merge_policy` must be one of: ['add', 'cat'], got {}".format(self.policy)
                        )

            class FPNDecoder(nn.Module):
                def __init__(
                        self,
                        encoder_channels=3,
                        encoder_depth=5,
                        pyramid_channels=256,
                        segmentation_channels=128,
                        dropout=0.2,
                        merge_policy="add",
                ):
                    super().__init__()

                    self.out_channels = segmentation_channels if merge_policy == "add" else segmentation_channels * 4
                    if encoder_depth < 3:
                        raise ValueError(
                            "Encoder depth for FPN decoder cannot be less than 3, got {}.".format(encoder_depth))

                    encoder_channels = encoder_channels
                    # encoder_channels = encoder_channels

                    self.p5 = nn.Conv2d(encoder_channels, pyramid_channels, kernel_size=1)
                    self.p4 = FPNBlock(pyramid_channels, encoder_channels)
                    self.p3 = FPNBlock(pyramid_channels, encoder_channels)
                    self.p2 = FPNBlock(pyramid_channels, encoder_channels)

                    self.seg_blocks = nn.ModuleList([
                        SegmentationBlock(pyramid_channels, segmentation_channels, n_upsamples=n_upsamples)
                        for n_upsamples in [3, 2, 1, 0]
                    ])

                    self.merge = MergeBlock(merge_policy)
                    self.dropout = nn.Dropout2d(p=dropout, inplace=True)

                def forward(self, features):
                    c2, c3, c4, c5 = features

                    p5 = self.p5(c5)
                    p4 = self.p4(p5, c4)
                    p3 = self.p3(p4, c3)
                    p2 = self.p2(p3, c2)

                    feature_pyramid = [seg_block(p) for seg_block, p in zip(self.seg_blocks, [p5, p4, p3, p2])]
                    x = self.merge(feature_pyramid)
                    x = self.dropout(x)

                    return x.permute(3, 2, 1, 0)

            #####




import torch
import torch.nn as nn
from collagen.core import Module
from collagen.modelzoo.segmentation import backbones
from collagen.modelzoo.segmentation import constants
from collagen.modelzoo.segmentation import decoders


class EncoderDecoder(Module):
    def __init__(self, n_outputs, backbone: str or Module, decoder: str or Module,
                 decoder_normalization='BN', spatial_dropout=None, bayesian_dropout=None,
                 unet_activation='relu', unet_width=32):
        super(EncoderDecoder, self).__init__()
        if isinstance(backbone, str):
            if backbone in constants.allowed_encoders:
                if 'resnet' in backbone:
                    backbone = backbones.ResNetBackbone(backbone, dropout=bayesian_dropout)
                else:
                    ValueError('Cannot find the implementation of the backbone!')
            else:
                raise ValueError('This backbone name is not in the list of allowed backbones!')

        if isinstance(decoder, str):
            if decoder in constants.allowed_decoders:
                if decoder == 'FPN':
                    decoder = decoders.FPNDecoder(encoder_channels=backbone.output_shapes,
                                                  pyramid_channels=256, segmentation_channels=128,
                                                  final_channels=n_outputs, spatial_dropout=spatial_dropout,
                                                  normalization=decoder_normalization,
                                                  bayesian_dropout=bayesian_dropout)
                elif decoder == 'UNet':
                    decoder = decoders.UNetDecoder(encoder_channels=backbone.output_shapes,
                                                   width=unet_width, activation=unet_activation,
                                                   final_channels=n_outputs,
                                                   spatial_dropout=spatial_dropout,
                                                   normalization=decoder_normalization)

        decoder.initialize()

        self.backbone = backbone
        self.decoder = decoder

    def forward(self, x):
        features = self.backbone(x)
        return self.decoder(features)

    def switch_dropout(self):
        """
        Has effect only if the model supports monte-carlo dropout inference.

        """
        self.backbone.switch_dropout()
        self.decoder.switch_dropout()

    def get_features(self):
        pass

    def get_features_by_name(self, name: str):
        pass

    class SimpleNet(nn.Module):
        def __init__(self,
                     num_classes=10,
                     final_channels=3):
            super(SimpleNet, self).__init__()

            self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, stride=1, padding=1)
            self.relu1 = nn.ReLU()

            self.conv2 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, stride=1, padding=1)
            self.relu2 = nn.ReLU()

            # self.pool = nn.MaxPool2d(kernel_size=2)

            self.conv3 = nn.Conv2d(in_channels=24, out_channels=24, kernel_size=3, stride=1, padding=1)
            self.relu3 = nn.ReLU()

            self.conv4 = nn.Conv2d(in_channels=24, out_channels=24, kernel_size=3, stride=1, padding=1)
            self.relu4 = nn.ReLU()

            self.final_layer = nn.Conv2d(in_channels=24, out_channels=final_channels, kernel_size=3, stride=1,
                                         padding=1)

            self.fc = nn.Linear(in_features=16 * 16 * 24, out_features=num_classes)

        def forward(self, input, out_shape=1024):
            output = self.conv1(input)
            output = self.relu1(output)

            output = self.conv2(output)
            output = self.relu2(output)

            #    output = self.pool(output)  # Max pooling

            output = self.conv3(output)
            output = self.relu3(output)

            output = self.conv4(output)
            output = self.relu4(output)

            output = F.interpolate(output, scale_factor=2, mode='bilinear', align_corners=True)  # Could be 'bilinear'

            output = self.final_layer(output)

            # output = output.view(-1, 16 * 16 * 24)

            # output = self.fc(output)
            if out_shape is not None:
                return F.interpolate(output, size=out_shape, mode='bilinear', align_corners=True)
            # else:
            # return output
            x = (10, 20, 30, 40, 50)
            for var in x:
                print("index " + str(x.index(var)) + ":", var)
fname_01=(1,3,2,2,4,5,5,9)
mask_fname_01=(1,3,2,2,4,5,5,9, 3, 4, 6)
df_frame = {'fname': [], 'mask_fname': []}
for i in range(fname_01):
    df_frame[i]= fname_01[i]


    images_repeat = []
    for i in range(len(images)):
       hmds_masks = glob(str(masks_loc) + '/*' + str(images[i])[:-13] + '**/*.png', recursive=True)
       images_repeat.extend(list(np.repeat(images[i], len(hmds_masks) // 60 + len(hmds_masks) % 60)))
       '''
       images_repeat = []
        for i in range(len(images)):
           hmds_masks = glob(str(masks_loc) + '/*' + str(images[i])[:-13] + '**/*.png', recursive=True)
           #if '6068-4L_00000039.png' in str(images[i]):
            #   images_repeat.extend(list(np.repeat(images[i], len(hmds_masks))))
        #   if '6069-4L' in str(images[i]):
         #      images_repeat.extend(list(np.repeat(images[i], len(hmds_masks))))
           if i in str(images[i]):
               images_repeat.extend(list(np.repeat(images[i], len(hmds_masks)*19)))
          # elif'.png' in str(images[i]):
           #     images_repeat.append(list(np.repeat(images[i], len(hmds_masks)//2 + len(hmds_masks) % 2)))

           # elif 'a.mat.png' in str(masks[i]):
            #    masks_repeat.extend(list(np.repeat(masks[i], len(hmds_images) // 2)))
            #elif '5928-4L_00000040.png' in str(masks[i]):
             #   masks_repeat.extend(list(np.repeat(images[i], len(masks))))
         '''


       def parse_grayscale(root, entry, transform, data_key, target_key, mean=False):
           # TODO make sure that this is working
           # Image and mask generation
           img = cv2.imread(str(entry.fname))
           img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
           # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
           # img = np.expand_dims(img, 1)
           img = np.transpose(img, (1, 0, 2))

           mask = cv2.imread(str(entry.mask_fname))
           # mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
           # mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

           try:
               mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
               mask[1, :, :] = mask[0, :, :]
               mask[2, :, :] = mask[0, :, :]
               mask = np.transpose(mask, (1, 0, 2))
           except Exception:
               print(str(entry.mask_fname))
           # mask = np.expand_dims(mask, 0)

           if img.shape[0] != mask.shape[0]:
               img = cv2.resize(img, (mask.shape[1], mask.shape[0]))
           elif img.shape[1] != mask.shape[1]:
               mask = mask[:, :img.shape[1]]

           # img = np.expand_dims(img, 1)
           # img=np.transpose(img,(0,2,1))
           img, mask = transform((img, mask))
           img = img.permute(2, 0, 1)  # / 255.  # img.shape[0] is the color channel after permute
           mask = mask.permute(2, 0, 1)
           # mask = ndimage.zoom(mask, 1.0)
           # img = ndimage.zoom(img, 1.0)

           # plt.imshow(mask[0,:])
           # plt.show()

           # plt.imshow(img[0,:])
           # plt.show()

           # Debugging
           # plt.imshow(np.asarray(img).transpose((1, 2, 0)))
           # plt.imshow(np.asarray(mask).squeeze(), alpha=0.3)
           # plt.show()

           # Calculate mean profile
           if mean:
               masks_repeat = []
               for i in range(len(masks)):
                   hmds_images = glob(str(images_loc) + '/*' + str(masks[i])[:-9] + '/*.png', recursive=True)
                   if '6061-12La.mat.png' in str(masks[i]):
                       masks_repeat.extend(list(np.repeat(masks[i], len(hmds_images))))
                   elif 'a.mat.png' in str(masks[i]):
                       masks_repeat.extend(list(np.repeat(masks[i], len(hmds_images) // 2)))
                   elif 'b.mat.png' in str(masks[i]):
                       masks_repeat.extend(list(np.repeat(masks[i], len(hmds_images) // 2 + len(hmds_images) % 2)))
                   else:
                       raise Exception('Wrong file name for the PLM image')

                   fig = plt.figure()
                   plot = fig.add_subplot(1, 3, 1)
                   plot.set_title('Initial Weights')
                   imgplot = plt.imshow(init_weights.view(-1, 280, 280).cpu()[0], cmap='gray')

                   plot = fig.add_subplot(1, 3, 2)
                   plot.set_title('Trained Weights')
                   imgplot = plt.imshow(trained_weights.view(-1, 280, 280).cpu()[0], cmap='gray')

                   plot = fig.add_subplot(1, 3, 3)
                   plot.set_title('Difference Weights')






import cv2
import numpy as np
import gc
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from pathlib import Path
import argparse
import dill
#from torch2trt import torch2trt
import torch
import torch.nn as nn
import yaml
from time import sleep, time
from tqdm import tqdm
from glob import glob
#from collagen.modelzoo.segmentation import EncoderDecoder
#from hmdscollagen.training.models import SimpleNet
from hmdscollagen.training.SimpleConvNet import SimpleConvNet

from collagen.core.utils import auto_detect_device

from hmdscollagen.data.utilities import load, save, print_orthogonal
from hmdscollagen.data.visualizations import render_volume

from pytorch_toolbelt.inference.tiles import ImageSlicer, CudaTileMerger
from pytorch_toolbelt.utils.torch_utils import tensor_from_rgb_image, to_numpy

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)


class InferenceModel(nn.Module):
    def __init__(self, models_list):
        super(InferenceModel, self).__init__()
        self.n_folds = len(models_list)
        modules = {}
        for idx, m in enumerate(models_list):
            modules[f'fold_{idx}'] = m

        self.__dict__['_modules'] = modules

    def forward(self, x):
        res = 0
        for idx in range(self.n_folds):
            fold = self.__dict__['_modules'][f'fold_{idx}']
            #res += torch2trt(fold, [x]).sigmoid()
            res += fold(x).sigmoid()

        return res / self.n_folds


def inference(inference_model, img_full, device='cuda'):
    x, y, ch = img_full.shape

    input_x = config['training']['crop_size'][0]
    input_y = config['training']['crop_size'][1]

    # Cut large image into overlapping tiles
    tiler = ImageSlicer(img_full.shape, tile_size=(input_x, input_y),
                        tile_step=(input_x // 2, input_y // 2), weight=args.weight)

    # HCW -> CHW. Optionally, do normalization here
    tiles = [tensor_from_rgb_image(tile) for tile in tiler.split(img_full)]

    # Allocate a CUDA buffer for holding entire mask
    merger = CudaTileMerger(tiler.target_shape, channels=1, weight=tiler.weight)

    # Run predictions for tiles and accumulate them
    for tiles_batch, coords_batch in DataLoader(list(zip(tiles, tiler.crops)), batch_size=args.bs, pin_memory=True):
        # Move tile to GPU
        tiles_batch = (tiles_batch.float() / 255.).to(device)
        # Predict and move back to CPU
        pred_batch = inference_model(tiles_batch)

        # Merge on GPU
        merger.integrate_batch(pred_batch, coords_batch)

        # Plot
        if args.plot:
            for i in range(args.bs):
                if args.bs != 1:
                    plt.imshow(pred_batch.cpu().detach().numpy().astype('float32').squeeze()[i, :, :])
                else:
                    plt.imshow(pred_batch.cpu().detach().numpy().astype('float32').squeeze())
                plt.show()

    # Normalize accumulated mask and convert back to numpy
    merged_mask = np.moveaxis(to_numpy(merger.merge()), 0, -1).astype('float32')
    merged_mask = tiler.crop_to_orignal_size(merged_mask)
    # Plot
    if args.plot:
        for i in range(args.bs):
            if args.bs != 1:
                plt.imshow(merged_mask)
            else:
                plt.imshow(merged_mask.squeeze())
            plt.show()

    torch.cuda.empty_cache()
    gc.collect()

    return merged_mask.squeeze()


if __name__ == "__main__":
    start = time()

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', type=Path, default='/data/Repositories/HMDS_orientation/Data/test/hmds_test_flip/')
    #parser.add_argument('--dataset_root', type=Path, default='/media/santeri/Transcend1/Full samples/')
    parser.add_argument('--save_dir', type=Path, default='/data/Repositories/HMDS_collagen/workdir/')
    parser.add_argument('--subdir', type=Path, choices=['NN_prediction', ''], default='')
    #parser.add_argument('--dataset_root', type=Path, default='../../../Data/µCT/images')
    #parser.add_argument('--save_dir', type=Path, default='/data/Repositories/HMDS_collagen/workdir/')
    parser.add_argument('--bs', type=int, default=300)
    parser.add_argument('--plot', type=bool, default=False)
    parser.add_argument('--weight', type=str, choices=['pyramid', 'mean'], default='mean')
    parser.add_argument('--completed', type=int, default=13)
    parser.add_argument('--snapshot', type=Path,
                        default='/data/Repositories/HMDS_collagen/workdir/snapshots/mipt-stud-dl-b_2020_06_03_07_41_22/')
    parser.add_argument('--dtype', type=str, choices=['.bmp', '.png', '.tif'], default='.png')
    args = parser.parse_args()
    subdir = ''  # 'NN_prediction'


    # Load snapshot configuration
    with open(args.snapshot / 'config.yml', 'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)

    with open(args.snapshot / 'args.dill', 'rb') as f:
        args_experiment = dill.load(f)

   # with open(args.snapshot / 'split_config.dill', 'rb') as f:
    #    split_config = dill.load(f)
    args.save_dir.mkdir(exist_ok=True)


    # Load models
    models = glob(str(args.snapshot) + '/*.pth')
    #models = glob(str(args.snapshot) + '/*fold_3_*.pth')
    models.sort()
    #device = auto_detect_device()
    device = 'cuda'  # Use the second GPU for inference

    # List the models
    model_list = []
    for fold in range(len(models)):
        #model = EncoderDecoder(**config['model'])
        model = SimpleConvNet().to(device)
        model.load_state_dict(torch.load(models[fold]))
        model_list.append(model)

    model = InferenceModel(model_list).to(device)
    #if torch.cuda.device_count() > 1:  # Multi-GPU
    #    model = nn.DataParallel(model).to(device)
    model.eval()

    threshold = 0.5 if config['training']['log_jaccard'] is False else 0.3  # Set probability threshold
    print(f'Found {len(model_list)} models.')

    # Load samples
    # samples = [os.path.basename(x) for x in glob(str(args.dataset_root / '*XZ'))]  # Load with specific name
    samples = os.listdir(args.dataset_root)
    samples.sort()
    # samples = [samples[id] for id in [7, 11]]  # Get intended samples from list

    # Skip the completed samples
   # if args.completed > 0:
    #    samples = samples[args.completed:]
    for idx, sample in enumerate(samples):
        #try:
        print(f'==> Processing sample {idx + 1} of {len(samples)}: {sample}')

        # Load image stacks
        data_xz, files = load(str(args.dataset_root / sample), rgb=True)
        data_xz = np.transpose(data_xz, (0, 1, 2,3)) # X-Z-Y-Ch
        data_yz = np.transpose(data_xz, (1, 2, 0, 3))  # Y-Z-X-Ch

        data_xz = data_xz[:, :, :, 0]
        data_xz = np.expand_dims(data_xz, axis=3)

        data_yz = data_yz[:, :, :, 0]
        data_yz = np.expand_dims(data_yz, 3)

        mask_xz = np.zeros(data_xz.shape)
        mask_yz = np.zeros(data_yz.shape)
        #mask_lz=(np.transpose(mask_yz, (0, 2, 1, 3))/2)

        # Loop for image slices
        # 1st orientation
        with torch.no_grad():  # Do not update gradients
            for slice_idx in tqdm(range(data_xz.shape[2]), desc='Running inference, XZ'):
                mask_xz[:, :, slice_idx, 0] = inference(model, data_xz[:, :, slice_idx, :])

                #plt.imshow(mask_xz[:, :,:,:])
                #plt.show()

            # 2nd orientation
           # for slice_idx in tqdm(range(data_yz.shape[2]), desc='Running inference, YZ'):
            #    mask_yz[:, :, slice_idx, 0] = inference(model, data_yz[:, :, slice_idx, :])

        # Average probability maps
        #mask_final = ((mask_xz + np.transpose(mask_yz, (0, 2, 1))) / 2) >= threshold
       # mask_final = ((mask_xz + np.transpose(mask_yz, (0, 2, 1, 3))) / 2)*255
        #k=np.transpose(mask_yz, (1,0, 2, 3))
        #mask_final = (mask_xz + np.transpose(mask_yz, (1,0, 2, 3)) / 2) * 255
       # mask_xz = list()
        #mask_yz = list()
        #data_xz = list()

        # Convert to original orientation
        #mask_final = np.transpose(mask_final, (0, 1, 2)).astype('uint8')#*255
        #mask_final = np.transpose(mask_final, (0, 2, 1))

        # Save predicted full mask
        if str(args.subdir) != '.':  # Save in original location
            save(str(args.dataset_root / sample / subdir), files, (mask_xz * 255).astype(np.uint16), dtype=args.dtype)
        else:  # Save in new location
            save(str(args.save_dir / sample), files, (mask_xz * 255).astype(np.uint16), dtype=args.dtype)

       # render_volume(data_yz[:, :, 0] * mask_final,
        #              savepath=str(args.save_dir / 'visualizations' / (sample + '_render' + args.dtype)),
         #             white=True, use_outline=False)

        print_orthogonal((mask_xz.squeeze() * 255).astype(np.uint16), invert=False, res=3.2, title=None, cbar=True,
                         savepath=str(args.save_dir / 'visualizations' / (sample + '_prediction.png')),
                         scale_factor=1000)
        #except Exception as e:
        #    print(f'Sample {sample} failed due to error:\n\n {e}\n\n.')
        #    continue
    dur = time() - start
    print(f'Inference completed in {dur // 3600} hours, {(dur % 3600) // 60} minutes, {dur % 60} seconds.')
