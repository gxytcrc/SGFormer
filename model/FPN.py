from torch import nn
import torch
import pdb

class FPNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(FPNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)

class FPN(nn.Module):
    def __init__(self, in_channels_list, out_channels, extra_blocks=None):
        super(FPN, self).__init__()
        self.lateral_layers = nn.ModuleList()
        self.pyramid_layers = nn.ModuleList()
        self.level_num = len(in_channels_list)
        # print("level_num: ", self.level_num)
        for in_channels in in_channels_list:
            self.lateral_layers.append(FPNBlock(in_channels, out_channels, 1, 1, 0))
            self.pyramid_layers.append(FPNBlock(out_channels, out_channels, 3, 1, 1))
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, input_features):

        lateral_features = [lateral(input_features[i]) for i, lateral in enumerate(self.lateral_layers)]
        pyramid_features = []
        p = lateral_features[-1]
        last_feature = self.pyramid_layers[-1](p)
        pyramid_features.append(last_feature)
        extra_feature = nn.functional.max_pool2d(last_feature, 1, 2, 0)
        # pyramid_features.append(extra_feature)
        for i in range(self.level_num - 2, -1, -1):
            size_i = lateral_features[i].shape[2:]
            p = lateral_features[i] + nn.functional.interpolate(p, size=size_i, mode='nearest')
            p = self.pyramid_layers[i](p)
            pyramid_features.insert(0, p)
        return pyramid_features

class FeaturePyramidUpsample(nn.Module):
    def __init__(self, input_dims, output_dim, target_size=(512, 512)):
        super(FeaturePyramidUpsample, self).__init__()
        self.target_size = target_size
        self.mlp = nn.Sequential(
            nn.Linear(sum(input_dims), output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim)
        )

    def forward(self, feature_pyramid):
        upsampled_features = []

        for feature in feature_pyramid:
            upsampled_feature = nn.functional.interpolate(feature, size = self.target_size)
            upsampled_features.append(upsampled_feature)
        concatenated_features = torch.cat(upsampled_features, dim=1)
        batch_size, dims, height, width = concatenated_features.size()
        concatenated_features = concatenated_features.view(batch_size, dims,-1)
        concatenated_features = concatenated_features.permute(0, 2, 1)
        output_features = self.mlp(concatenated_features)
        output_features = output_features.permute(0, 2, 1)
        output_features = output_features.view(batch_size, -1, height, width)
        return output_features


class ConvModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0):
        super(ConvModule, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        return self.conv(x)


class FPNNeck(nn.Module):
    def __init__(self, out_dim):
        super(FPNNeck, self).__init__()

        # Lateral convolution layer
        self.lateral_convs = nn.ModuleList([
            ConvModule(1024, out_dim, kernel_size=1, stride=1)
        ])

        # FPN convolution layer
        self.fpn_convs = nn.ModuleList([
            ConvModule(out_dim, out_dim, kernel_size=3, stride=1, padding=1)
        ])

        # Initialize weights
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, input_features):
        # Apply lateral convolutions
        lateral_features = [lateral(input_features[i]) for i, lateral in enumerate(self.lateral_convs)]

        # Apply FPN convolutions
        fpn_out = [fpn_conv(lateral_features[i]) for i, fpn_conv in enumerate(self.fpn_convs)]

        return fpn_out