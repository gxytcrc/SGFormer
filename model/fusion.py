import torch
import torch.nn as nn
import torch.nn.functional as F
from .transformer.resnet_3d import BasicBlock3D, SeparableConv3D
from .resnet import BasicBlock
import pdb

class ChannelAttentionModule(nn.Module):
    def __init__(self, cin, cout, reduction=16):
        super(ChannelAttentionModule, self).__init__()
        mid_channel = cin // reduction
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)

        self.shared_MLP = nn.Sequential(
            nn.Linear(in_features=cin, out_features=mid_channel),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(in_features=mid_channel, out_features=cout)
        )
        # self.another_MLP = nn.Sequential(
        #     nn.Conv3d(in_channels=cin, out_channels=mid_channel, kernel_size=1),
        #     nn.LeakyReLU(0.1, inplace=True),
        #     nn.Conv3d(in_channels=mid_channel, out_channels=cout, kernel_size=1)
        # )

        # self.act = nn.Sigmoid()
        # self.act=nn.SiLU()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x).view(x.size(0), -1)).unsqueeze(2).unsqueeze(3).unsqueeze(4)
        maxout = self.shared_MLP(self.max_pool(x).view(x.size(0), -1)).unsqueeze(2).unsqueeze(3).unsqueeze(4)
        fusedout = avgout + maxout
        # cnnout = self.another_MLP(x)
        #self.act(fusedout + cnnout)
        return fusedout


class ChannelAttentionModule2d(nn.Module):
    def __init__(self, cin, cout, reduction=16):
        super(ChannelAttentionModule2d, self).__init__()
        mid_channel = cin // reduction
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Linear(in_features=cin, out_features=mid_channel),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(in_features=mid_channel, out_features=cout)
        )
        self.another_MLP = nn.Sequential(
            nn.Conv2d(in_channels=cin, out_channels=mid_channel, kernel_size=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channels=mid_channel, out_channels=cout, kernel_size=1)
        )

        self.act = nn.Sigmoid()
        # self.act=nn.SiLU()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x).view(x.size(0), -1)).unsqueeze(2).unsqueeze(3)
        maxout = self.shared_MLP(self.max_pool(x).view(x.size(0), -1)).unsqueeze(2).unsqueeze(3)
        fusedout = avgout + maxout
        cnnout = self.another_MLP(x)
        return self.act(fusedout + cnnout)

class SpatialAttentionModule(nn.Module):
    def __init__(self, cin):
        super(SpatialAttentionModule, self).__init__()
        self.conv3d = nn.Conv3d(in_channels=2, out_channels=1, kernel_size=7, padding=3, dilation=1)
        self.local_net = nn.Sequential(
            nn.Conv3d(in_channels=cin, out_channels=cin//2, kernel_size=3, padding=1, stride=1),
            BasicBlock3D(cin//2, cin//2),
            nn.Conv3d(in_channels=cin//2, out_channels=1, kernel_size=1)
        )
        self.act = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        global_out = self.conv3d(out)
        local_out = self.local_net(x)
        out = self.act(global_out + local_out)
        return out


class SpatialAttentionModule2d(nn.Module):
    def __init__(self, cin):
        super(SpatialAttentionModule2d, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        global_out = self.conv2d(out)
        return global_out


class HeightNet(nn.Module):
    def __init__(self, cin, height):
        super(HeightNet, self).__init__()
        self.height_net = nn.Sequential(
            nn.Conv2d(in_channels=cin, out_channels=cin, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=cin, out_channels=height, kernel_size=3, stride=1, padding=1)
        )
        self.act = nn.Sigmoid()
    def forward(self, x):
        x = self.height_net(x)
        return self.act(x)

class CBAM(nn.Module):
    def __init__(self, cin, cout, mode="3d"):
        super(CBAM, self).__init__()
        if mode=="3d":
            self.channel_attention = ChannelAttentionModule(cin, cout)
            self.spatial_attention = SpatialAttentionModule(cout)
        elif mode=="2d":
            self.channel_attention = ChannelAttentionModule2d(cin, cout)
            self.spatial_attention = SpatialAttentionModule2d(cout)

    def forward(self, x1, x2):
        x = torch.concat((x1, x2), dim=1)
        channel_weight = self.channel_attention(x)
        out = channel_weight * x1 + (1-channel_weight) * x2
        out = self.spatial_attention(out) * out
        return out



class CBAMV2(nn.Module):
    def __init__(self, cin, cout, mode="3d"):
        super(CBAMV2, self).__init__()
        self.channel_attention = ChannelAttentionModule(cin, cout)
        self.spatial_attention = SpatialAttentionModule2d(cin)
        self.spatial_refine = SpatialAttentionModule(cout)
        mid_channel = cin // 16
        self.another_MLP = nn.Sequential(
            nn.Conv3d(in_channels=cin, out_channels=mid_channel, kernel_size=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(in_channels=mid_channel, out_channels=cout, kernel_size=1)
        )
        self.act = nn.Sigmoid()


    def forward(self, vox, bev):
        vox_squeezed = torch.max_pool3d(kernel_size=(1, 1, 16), input=vox).squeeze(-1)
        x_0 = torch.concat((vox_squeezed, bev), dim=1)
        spatial_weight = 0.01 * self.spatial_attention(x_0).unsqueeze(-1).repeat(1, 1, 1, 1, 16)
        bev_lifted = bev.unsqueeze(-1).repeat(1, 1, 1, 1, 16)
        x_1 = torch.concat((vox, bev_lifted), dim=1)
        channel_weight = self.channel_attention(x_1)
        adaptive_weight = self.act(self.another_MLP(x_1)  + channel_weight + spatial_weight)
        fused = adaptive_weight * vox + (1-adaptive_weight) * bev_lifted
        out = self.spatial_refine(fused) * fused
        return out


class ChannelFusion(nn.Module):
    def __init__(self, cin, cout, mode="3d"):
        super(ChannelFusion, self).__init__()
        self.channel_attention = ChannelAttentionModule(cin, cout)

    def forward(self, x1, x2):
        x = torch.concat((x1, x2), dim=1)
        channel_weight = self.channel_attention(x)
        out = channel_weight * x1 + (1 - channel_weight) * x2
        return out

class SpatialFusion(nn.Module):
    def __init__(self, cin, cout, mode="3d"):
        super(SpatialFusion, self).__init__()
        self.spatial_attention = SpatialAttentionModule(cout)
        self.decoder = nn.Sequential(
            nn.Conv3d(cin, cout, kernel_size=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x1, x2):
        x = torch.concat((x1, x2), dim=1)
        x = self.decoder(x)
        out = self.spatial_attention(x) * x
        return out