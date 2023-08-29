import os
os.environ["TORCH_USE_CUDA_DSA"] = "1"
from torch import nn
import torch
from torchsummary import summary
import torch.nn.functional as F
from torch.nn.init import constant_

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Reshape(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)


class Generator(nn.Sequential):
    def __init__(self, gen_filters=250, filter_size=11, input_dim=50, img_rows = 240):
        super().__init__(
            nn.Linear(input_dim, gen_filters*15),
            nn.BatchNorm1d(gen_filters*15),
            nn.ReLU(True),
            Reshape((-1, gen_filters, 15)), # shape: (batch, channels, time)
            
            # Conv layer 1
            # In: 15 X 1 X 1, depth = 400
            # Out: 30 X 1 X 1, depth = 200
            nn.ConvTranspose1d(in_channels=gen_filters, out_channels=gen_filters//2,
                                kernel_size=filter_size, stride=2, padding=5, output_padding=1),
            nn.BatchNorm1d(gen_filters//2),
            nn.ReLU(True),

            nn.ConvTranspose1d(in_channels=gen_filters//2, out_channels=gen_filters//4,
                                kernel_size=filter_size, stride=2, padding = 5, output_padding=1),
            nn.BatchNorm1d(gen_filters//4),
            nn.ReLU(True),

            nn.ConvTranspose1d(in_channels=gen_filters//4, out_channels=gen_filters//8,
                                kernel_size=filter_size, stride=2, padding = 5, output_padding=1),
            nn.BatchNorm1d(gen_filters//8),
            nn.ReLU(True),


            nn.ConvTranspose1d(in_channels=gen_filters//8, out_channels=gen_filters//16,
                                kernel_size=filter_size, stride=2, padding = 5, output_padding=1),
            nn.BatchNorm1d(gen_filters//16),
            nn.ReLU(True),

            nn.ConvTranspose1d(in_channels=gen_filters//16, out_channels=1, 
                               kernel_size=filter_size, stride = 1, padding=5),
            nn.Sigmoid(),
            Reshape((-1, img_rows))

        )
        


class Discriminator(nn.Sequential):
    def __init__(self, img_rows=240, dropout=0.4, dis_filters=20, filter_size=11, input_dim =50):
        super().__init__(
            Reshape((-1, 1, img_rows)), # shape: (batch, channels, time)
            # Layer 1
            # In: 240 X 1 X 1, depth = 1
            # Out: 120 X 1 X 1, depth = 25
            nn.Conv1d(
                in_channels=1,
                out_channels=dis_filters,
                kernel_size=filter_size,
                stride=2, padding = 5),
            nn.BatchNorm1d(dis_filters),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),

            # Layer 2
            # In: 120 X 1 X 1, depth = 25
            # Out: 60 X 1 X 1, depth = 50
            nn.Conv1d(
                in_channels=dis_filters,
                out_channels=dis_filters*2,
                kernel_size=filter_size,
                stride=2, padding = 5),
            nn.BatchNorm1d(dis_filters*2),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),

            # Layer 3
            # In: 60 X 1 X 1, depth = 50
            # Out: 30 X 1 X 1, depth = 100
            nn.Conv1d(
                in_channels=dis_filters*2,
                out_channels=dis_filters*4,
                kernel_size=filter_size,
                stride=2, padding = 5),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(dis_filters*4),
            nn.Dropout(dropout),

            # Layer 4
            # In: 30 X 1 X 1, depth = 100
            # Out: 15 X 1 X 1, depth = 200
            nn.Conv1d(
                in_channels=dis_filters*4,
                out_channels=dis_filters*8,
                kernel_size=filter_size,
                stride=2, padding = 5),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(dis_filters*8),
            nn.Dropout(dropout),

            # # Output layer
            nn.Flatten(),
            nn.Linear(15*dis_filters*8, 1),
            nn.Sigmoid(),
            
        )

        


