import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
import numpy as np

class SegmentationHead(nn.Module):
  '''
  3D Segmentation heads to retrieve semantic segmentation at each scale.
  Formed by Dim expansion, Conv3D, ASPP block, Conv3D.
  '''
  def __init__(self, inplanes, planes, nbr_classes, dilations_conv_list):
    super().__init__()

    # First convolution
    self.conv0 = nn.Conv3d(inplanes, planes, kernel_size=3, padding=1, stride=1)

    # ASPP Block
    self.conv_list = dilations_conv_list
    self.conv1 = nn.ModuleList(
      [nn.Conv3d(planes, planes, kernel_size=3, padding=dil, dilation=dil, bias=False) for dil in dilations_conv_list])
    self.bn1 = nn.ModuleList([nn.BatchNorm3d(planes) for dil in dilations_conv_list])
    self.conv2 = nn.ModuleList(
      [nn.Conv3d(planes, planes, kernel_size=3, padding=dil, dilation=dil, bias=False) for dil in dilations_conv_list])
    self.bn2 = nn.ModuleList([nn.BatchNorm3d(planes) for dil in dilations_conv_list])
    self.relu = nn.ReLU(inplace=True)

    # Convolution for output
    self.conv_classes = nn.Conv3d(planes, nbr_classes, kernel_size=3, padding=1, stride=1)

  def forward(self, x_in):

    # Dimension exapension
    x_in = x_in[:, None, :, :, :]

    # Convolution to go from inplanes to planes features...
    x_in = self.relu(self.conv0(x_in))

    y = self.bn2[0](self.conv2[0](self.relu(self.bn1[0](self.conv1[0](x_in)))))
    for i in range(1, len(self.conv_list)):
      y += self.bn2[i](self.conv2[i](self.relu(self.bn1[i](self.conv1[i](x_in)))))
    x_in = self.relu(y + x_in)  # modified

    x_in = self.conv_classes(x_in)

    return x_in

class VoxelModel(nn.Module):
    def __init__(self, args):  # device='cuda:0',
        super(VoxelModel, self).__init__()

        self.nbr_classes = 2
        self.gamma = 0
        self.alpha = 0.54
        self.input_dimensions = [256, 32, 256]  # Grid dimensions should be (W, H, D).. z or height being axis 1
        f = self.input_dimensions[1]
        self.pool = nn.MaxPool2d(2)  # [F=2; S=2; P=0; D=1]
        self.class_frequencies_level1 =  np.array([5.41773033e09, 4.03113667e08])
        self.class_weights_level_1 = torch.from_numpy(
            1 / np.log(self.class_frequencies_level1 + 0.001)
        )
        self.pooling = nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2))
        self.out_scale ='1_2'
        self.Encoder_block1 = nn.Sequential(
            nn.Conv2d(f, f, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(f, f, kernel_size=3, padding=1, stride=1),
            nn.ReLU()
        )

        self.Encoder_block2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(f, int(f * 1.5), kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(int(f * 1.5), int(f * 1.5), kernel_size=3, padding=1, stride=1),
            nn.ReLU()
        )

        self.Encoder_block3 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(int(f * 1.5), int(f * 2), kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(int(f * 2), int(f * 2), kernel_size=3, padding=1, stride=1),
            nn.ReLU()
        )

        self.Encoder_block4 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(int(f * 2), int(f * 2.5), kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(int(f * 2.5), int(f * 2.5), kernel_size=3, padding=1, stride=1),
            nn.ReLU()
        )

        # Treatment output 1:8
        self.conv_out_scale_1_8 = nn.Conv2d(int(f * 2.5), int(f / 8), kernel_size=3, padding=1, stride=1)
        self.deconv_1_8__1_2 = nn.ConvTranspose2d(int(f / 8), int(f / 8), kernel_size=4, padding=0, stride=4)
        self.deconv_1_8__1_1 = nn.ConvTranspose2d(int(f / 8), int(f / 8), kernel_size=8, padding=0, stride=8)

        # Treatment output 1:4
        if self.out_scale == "1_4" or self.out_scale == "1_2" or self.out_scale == "1_1":
            self.deconv1_8 = nn.ConvTranspose2d(int(f / 8), int(f / 8), kernel_size=6, padding=2, stride=2)
            self.conv1_4 = nn.Conv2d(int(f * 2) + int(f / 8), int(f * 2), kernel_size=3, padding=1, stride=1)
            self.conv_out_scale_1_4 = nn.Conv2d(int(f * 2), int(f / 4), kernel_size=3, padding=1, stride=1)
            self.deconv_1_4__1_1 = nn.ConvTranspose2d(int(f / 4), int(f / 4), kernel_size=4, padding=0, stride=4)

        # Treatment output 1:2
        if self.out_scale == "1_2" or self.out_scale == "1_1":
            self.deconv1_4 = nn.ConvTranspose2d(int(f / 4), int(f / 4), kernel_size=6, padding=2, stride=2)
            self.conv1_2 = nn.Conv2d(int(f * 1.5) + int(f / 4) + int(f / 8), int(f * 1.5), kernel_size=3, padding=1,
                                     stride=1)
            self.conv_out_scale_1_2 = nn.Conv2d(int(f * 1.5), int(f / 2), kernel_size=3, padding=1, stride=1)

        # Treatment output 1:1
        if self.out_scale == "1_1":
            self.deconv1_2 = nn.ConvTranspose2d(int(f / 2), int(f / 2), kernel_size=6, padding=2, stride=2)
            self.conv1_1 = nn.Conv2d(int(f / 8) + int(f / 4) + int(f / 2) + int(f), f, kernel_size=3, padding=1,
                                     stride=1)

        if self.out_scale == "1_1":
            self.seg_head_1_1 = SegmentationHead(1, 8, self.nbr_classes, [1, 2, 3])
        elif self.out_scale == "1_2":
            self.seg_head_1_2 = SegmentationHead(1, 8, self.nbr_classes, [1, 2, 3])
        elif self.out_scale == "1_4":
            self.seg_head_1_4 = SegmentationHead(1, 8, self.nbr_classes, [1, 2, 3])
        elif self.out_scale == "1_8":
            self.seg_head_1_8 = SegmentationHead(1, 8, self.nbr_classes, [1, 2, 3])

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def step(self, input):
        # input = x['3D_OCCUPANCY']  # Input to LMSCNet model is 3D occupancy big scale (1:1) [bs, 1, W, H, D]
        # input = torch.squeeze(input, dim=1).permute(0, 2, 1, 3)  # Reshaping to the right way for 2D convs [bs, H, W, D]

        # print(input.shape) [4, 32, 256, 256]
        # Encoder block
        _skip_1_1 = self.Encoder_block1(input)
        # print('_skip_1_1.shape', _skip_1_1.shape)  # [1, 32, 256, 256]
        _skip_1_2 = self.Encoder_block2(_skip_1_1)
        # print('_skip_1_2.shape', _skip_1_2.shape)  # [1, 48, 128, 128]
        _skip_1_4 = self.Encoder_block3(_skip_1_2)
        # print('_skip_1_4.shape', _skip_1_4.shape)  # [1, 64, 64, 64]
        _skip_1_8 = self.Encoder_block4(_skip_1_4)
        # print('_skip_1_8.shape', _skip_1_8.shape)  # [1, 80, 32, 32]

        # cross_1_1 = self.Encoder_1(cross_input)
        # cross_1_2 = self.Encoder_2(cross_1_1)
        # cross_1_4 = self.Encoder_3(cross_1_2)
        # cross_1_8 = self.Encoder_block4(cross_1_4)

        # Out 1_8
        out_scale_1_8__2D = self.conv_out_scale_1_8(_skip_1_8)

        # print('out_scale_1_8__2D.shape', out_scale_1_8__2D.shape)  # [1, 4, 32, 32]

        if self.out_scale == "1_8":
            out_scale_1_8__3D = self.seg_head_1_8(out_scale_1_8__2D)  # [1, 20, 16, 128, 128]
            out_scale_1_8__3D = out_scale_1_8__3D.permute(0, 1, 3, 4, 2)  # [1, 20, 128, 128, 16]
            return out_scale_1_8__3D

        elif self.out_scale == "1_4":
            # Out 1_4
            out = self.deconv1_8(out_scale_1_8__2D)
            out = torch.cat((out, _skip_1_4), 1)
            out = F.relu(self.conv1_4(out))
            out_scale_1_4__2D = self.conv_out_scale_1_4(out)

            out_scale_1_4__3D = self.seg_head_1_4(out_scale_1_4__2D)  # [1, 20, 16, 128, 128]
            out_scale_1_4__3D = out_scale_1_4__3D.permute(0, 1, 3, 4, 2)  # [1, 20, 128, 128, 16]
            return out_scale_1_4__3D

        elif self.out_scale == "1_2":
            # Out 1_4
            out = self.deconv1_8(out_scale_1_8__2D)
            out = torch.cat((out, _skip_1_4), 1)
            out = F.relu(self.conv1_4(out))
            out_scale_1_4__2D = self.conv_out_scale_1_4(out)

            # Out 1_2
            out = self.deconv1_4(out_scale_1_4__2D)
            out = torch.cat((out, _skip_1_2, self.deconv_1_8__1_2(out_scale_1_8__2D)), 1)
            out = F.relu(self.conv1_2(out))  # torch.Size([1, 48, 128, 128])
            out_scale_1_2__2D = self.conv_out_scale_1_2(out)  # torch.Size([1, 16, 128, 128])

            out_scale_1_2__3D = self.seg_head_1_2(out_scale_1_2__2D)  # [1, 20, 16, 128, 128]
            out_scale_1_2__3D = out_scale_1_2__3D.permute(0, 1, 3, 4, 2)  # [1, 20, 128, 128, 16]
            return out_scale_1_2__3D

        elif self.out_scale == "1_1":
            # Out 1_4
            out = self.deconv1_8(out_scale_1_8__2D)
            print('out.shape', out.shape)  # [1, 4, 64, 64]
            out = torch.cat((out, _skip_1_4), 1)
            out = F.relu(self.conv1_4(out))
            out_scale_1_4__2D = self.conv_out_scale_1_4(out)
            # print('out_scale_1_4__2D.shape', out_scale_1_4__2D.shape)  # [1, 8, 64, 64]

            # Out 1_2
            out = self.deconv1_4(out_scale_1_4__2D)
            print('out.shape', out.shape)  # [1, 8, 128, 128]
            out = torch.cat((out, _skip_1_2, self.deconv_1_8__1_2(out_scale_1_8__2D)), 1)
            out = F.relu(self.conv1_2(out))  # torch.Size([1, 48, 128, 128])
            out_scale_1_2__2D = self.conv_out_scale_1_2(out)  # torch.Size([1, 16, 128, 128])
            # print('out_scale_1_2__2D.shape', out_scale_1_2__2D.shape)  # [1, 16, 128, 128]

            # Out 1_1
            out = self.deconv1_2(out_scale_1_2__2D)
            out = torch.cat(
                (out, _skip_1_1, self.deconv_1_4__1_1(out_scale_1_4__2D), self.deconv_1_8__1_1(out_scale_1_8__2D)), 1)
            out_scale_1_1__2D = F.relu(self.conv1_1(out))  # [bs, 32, 256, 256]

            out_scale_1_1__3D = self.seg_head_1_1(out_scale_1_1__2D)
            # Take back to [W, H, D] axis order
            out_scale_1_1__3D = out_scale_1_1__3D.permute(0, 1, 3, 4, 2)  # [bs, C, H, W, D] -> [bs, C, W, H, D]
            return out_scale_1_1__3D
