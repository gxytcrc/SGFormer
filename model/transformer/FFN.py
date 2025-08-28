import torch
import torch.nn as nn

class FFN(nn.Module):
    def __init__(self, dims, in_dims):
        super(FFN, self).__init__()
        self.activate = nn.ReLU(inplace=True)
        self.layers = nn.Sequential(
            nn.Sequential(
                nn.Linear(in_features=dims, out_features=in_dims, bias=True),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.1, inplace=False)
            ),
            nn.Linear(in_features=in_dims, out_features=dims, bias=True),
            nn.Dropout(p=0.1, inplace=False)
        )
        self.dropout_layer = nn.Identity()

    def forward(self, x):
        x = self.activate(x)
        x = self.layers(x)
        x = self.dropout_layer(x)
        return x


class Unet(nn.Module):
    def __init__(self, input_dims, hidden_dims, output_dims):
        super(Unet, self).__init__()
        self.up_layer_1 = nn.Sequential(
            # nn.Conv3d(in_channels=hidden_dims, out_channels=hidden_dims, kernel_size=(3, 3, 1), stride=(2, 2, 1), padding=(1, 1, 0), bias=False),
            # nn.BatchNorm3d(hidden_dims),
            nn.ConvTranspose3d(in_channels=hidden_dims, out_channels=hidden_dims, kernel_size=(3, 3, 4),
                               padding=(1, 1, 0), stride=(1, 1, 1), bias=True),
            nn.BatchNorm3d(hidden_dims),
            nn.ReLU()
        )
        self.up_layer_2 = nn.Sequential(
            nn.Conv3d(in_channels=hidden_dims, out_channels=hidden_dims, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(hidden_dims),
            nn.ConvTranspose3d(in_channels=hidden_dims, out_channels=input_dims, kernel_size=(1, 1, 4),
                               padding=(0, 0, 0), stride=(1, 1, 2), dilation=(1, 1, 3), bias=True),
            nn.BatchNorm3d(input_dims),
            nn.ReLU()
        )
        # self.up_layer_3 = nn.Sequential(
        #     nn.ConvTranspose3d(in_channels=hidden_dims, out_channels=hidden_dims, kernel_size=(3, 3, 4),
        #                        padding=(1, 1, 0), stride=(1, 1, 1), bias=True),
        #     nn.BatchNorm3d(input_dims),
        #     nn.ReLU(),
        #     nn.Conv3d(in_channels=hidden_dims, out_channels=hidden_dims, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False),
        #     nn.BatchNorm3d(hidden_dims),
        #     nn.ReLU(),
        #     nn.ConvTranspose3d(in_channels=hidden_dims, out_channels=input_dims, kernel_size=(1, 1, 4),
        #                        padding=(0, 0, 0), stride=(1, 1, 2), dilation=(1, 1, 3), bias=True),
        #     nn.BatchNorm3d(input_dims),
        #     nn.ReLU()
        # )

    def upsample_1(self, input):
        output = self.up_layer_1(input)
        return output

class Decoder(nn.Module):
    def __init__(self, input_dims, hidden_dims, output_dims):
        super(Decoder, self).__init__()
        self.activate = nn.ReLU(inplace=True)
        self.layers = nn.Sequential(
            nn.Sequential(
                nn.Linear(in_features=input_dims, out_features=hidden_dims, bias=True),
                nn.ReLU(inplace=True),
            ),
            nn.Linear(in_features=hidden_dims, out_features=output_dims, bias=True),
        )
        self.dropout_layer = nn.Identity()

    def forward(self, x):
        x = self.activate(x)
        x = self.layers(x)
        return x
    
class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, bn_momentum):
        super(Upsample, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose3d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                dilation=1,
                output_padding=1,
            ),
            norm_layer(out_channels, momentum=bn_momentum),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.main(x)
