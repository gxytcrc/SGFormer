""" Full assembly of the parts to form the complete network """

from .unet_parts import *
import pdb

class UNet3D(nn.Module):
    def __init__(self, in_channels, out_channels, base_channels=128):
        super(UNet3D, self).__init__()

        # Encoder (3 layers)
        self.encoder1 = self.double_conv(in_channels, base_channels)
        self.encoder2 = self.double_conv(base_channels, base_channels * 2)
        self.encoder3 = self.double_conv(base_channels * 2, base_channels * 4)

        # Maxpooling layer
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = self.double_conv(base_channels * 4, base_channels * 8)

        # Decoder (3 layers)
        self.upconv3 = self.upconv(base_channels * 8, base_channels * 4)
        self.decoder3 = self.double_conv(base_channels * 8, base_channels * 4)
        self.upconv2 = self.upconv(base_channels * 4, base_channels * 2)
        self.decoder2 = self.double_conv(base_channels * 4, base_channels * 2)
        self.upconv1 = self.upconv(base_channels * 2, base_channels)
        self.decoder1 = self.double_conv(base_channels * 2, base_channels)

        # Output layer
        self.output_layer = nn.Conv3d(base_channels, out_channels, kernel_size=1)
        # self.conv_layer = nn.Conv3d(in_channels=1024, out_channels=1024, kernel_size=(1, 1, 2))

    def double_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )

    def upconv(self, in_channels, out_channels):
        return nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool(enc1))
        enc3 = self.encoder3(self.pool(enc2))

        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc3))
        # bottleneck_compressed = self.conv_layer(bottleneck)  # Shape [1, 1024, 16, 16, 1]
        # # Remove the last singleton dimension to get shape [1, 1024, 16, 16]
        # bottleneck_2d = bottleneck_compressed.squeeze(-1)  # Shape [1, 1024, 16, 16]
        # bottleneck_flattened = bottleneck_2d.flatten(2) # Shape: [1, 1024, 256]
        # bottleneck_flattened = bottleneck_flattened / bottleneck_flattened.norm(dim=1, keepdim=True)
        # Decoder
        dec3 = self.upconv3(bottleneck)
        dec3 = torch.cat((enc3, dec3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((enc2, dec2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((enc1, dec1), dim=1)
        dec1 = self.decoder1(dec1)

        return self.output_layer(dec1)
    
class SeparableConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(SeparableConv2D, self).__init__()
        # Depth-wise convolution
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding, groups=in_channels)
        # Point-wise convolution
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class UNet2D(nn.Module):
    def __init__(self, in_channels, out_channels, base_channels=128):
        super(UNet2D, self).__init__()
        dropout_rate = 0.1 
        # Encoder (3 layers)
        self.encoder1 = self.double_conv(in_channels, base_channels, dropout_rate)
        self.encoder2 = self.double_conv(base_channels, base_channels * 2, dropout_rate)
        self.encoder3 = self.double_conv(base_channels * 2, base_channels * 4, dropout_rate)

        # Maxpooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = self.double_conv(base_channels * 4, base_channels * 8, dropout_rate)

        # Decoder (3 layers)
        self.upconv3 = self.upconv(base_channels * 8, base_channels * 4)
        self.decoder3 = self.double_conv(base_channels * 8, base_channels * 4, dropout_rate)
        self.upconv2 = self.upconv(base_channels * 4, base_channels * 2)
        self.decoder2 = self.double_conv(base_channels * 4, base_channels * 2, dropout_rate)
        self.upconv1 = self.upconv(base_channels * 2, base_channels)
        self.decoder1 = self.double_conv(base_channels * 2, base_channels, dropout_rate)

        # Output layer
        self.output_layer = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def double_conv(self, in_channels, out_channels, dropout_rate):
        return nn.Sequential(
            SeparableConv2D(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            SeparableConv2D(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )

    def upconv(self, in_channels, out_channels):
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool(enc1))
        enc3 = self.encoder3(self.pool(enc2))

        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc3))
        # bottleneck_flattened = bottleneck.flatten(2) # Shape: [1, 1024, 256]
        # bottleneck_flattened = bottleneck_flattened / bottleneck_flattened.norm(dim=1, keepdim=True)
        # Decoder
        dec3 = self.upconv3(bottleneck)
        dec3 = torch.cat((enc3, dec3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((enc2, dec2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((enc1, dec1), dim=1)
        dec1 = self.decoder1(dec1)

        return self.output_layer(dec1)
        # return dec3, dec2, dec1, self.output_layer(dec1)


class UNetFusion(nn.Module):
    def __init__(self, in_channels, out_channels, base_channels=128):
        super(UNetFusion, self).__init__()
        self.unet_2d = UNet2D(in_channels, out_channels)
                # Encoder (3 layers)
        self.encoder1 = self.double_conv(in_channels, base_channels)
        self.encoder2 = self.double_conv(base_channels, base_channels * 2)
        self.encoder3 = self.double_conv(base_channels * 2, base_channels * 4)

        # Maxpooling layer
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = self.double_conv(base_channels * 4, base_channels * 8)

        # Decoder (3 layers)
        self.upconv3 = self.upconv(base_channels * 8, base_channels * 4)
        self.decoder3 = self.double_conv(base_channels * 4, base_channels * 4)
        self.upconv2 = self.upconv(base_channels * 4, base_channels * 2)
        self.decoder2 = self.double_conv(base_channels * 2, base_channels * 2)
        self.upconv1 = self.upconv(base_channels * 2, base_channels)
        self.decoder1 = self.double_conv(base_channels * 1, base_channels)

        #spatial fusion
        self.sp_attention_3 = SpatialFusion(base_channels * 8, base_channels * 4)
        self.sp_attention_2 = SpatialFusion(base_channels * 4, base_channels * 2)
        self.sp_attention_1 = SpatialFusion(base_channels * 2, base_channels)

        #channel fusion
        self.ch_attention_3 = ChannelFusion(base_channels * 8, base_channels * 4)
        self.ch_attention_2 = ChannelFusion(base_channels * 4, base_channels * 2)
        self.ch_attention_1 = ChannelFusion(base_channels * 2, base_channels)

        # Output layer
        self.output_layer = nn.Conv3d(base_channels, out_channels, kernel_size=1)
        # self.conv_layer = nn.Conv3d(in_channels=1024, out_channels=1024, kernel_size=(1, 1, 2))

    def upconv(self, in_channels, out_channels):
        return nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)

    def double_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )

    def forward(self, vox_features, bev_features):
        bev3, bev2, bev1, bev_output = self.unet_2d(bev_features)
        enc1 = self.encoder1(vox_features)
        enc2 = self.encoder2(self.pool(enc1))
        enc3 = self.encoder3(self.pool(enc2))

        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc3))

        # Decoder
        dec3 = self.upconv3(bottleneck)
        bev3 = bev3.unsqueeze(-1).repeat(1, 1, 1, 1, dec3.shape[-1])
        dec3 = self.sp_attention_3(dec3, bev3)
        dec3 = self.ch_attention_3(enc3, dec3)
        # dec3 = torch.cat((enc3, dec3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        bev2 = bev2.unsqueeze(-1).repeat(1, 1, 1, 1, dec2.shape[-1])
        dec2 = self.sp_attention_2(dec2, bev2)
        dec2 = self.ch_attention_2(enc2, dec2)
        # dec2 = torch.cat((enc2, dec2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        bev1 = bev1.unsqueeze(-1).repeat(1, 1, 1, 1, dec1.shape[-1])
        dec1 = self.sp_attention_1(dec1, bev1)
        dec1 = self.ch_attention_1(enc1, dec1)
        # dec1 = torch.cat((enc1, dec1), dim=1)
        dec1 = self.decoder1(dec1)
        return self.output_layer(dec1), bev_output
