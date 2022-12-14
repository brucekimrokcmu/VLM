# https://arxiv.org/pdf/2109.12098.pdf
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ConvBlock(nn.Module):
    def __init__(self, channels_in, channels_out, stride, unique_last_dim = False, residual=False, output_act=nn.ReLU):
        super().__init__()
        self.residual = residual
        # out = conv, conv, conv on input
        self.conv1 = nn.Conv2d(channels_in, channels_out, kernel_size=1, stride=1, bias=False)
        self.conv2 = nn.Conv2d(channels_out, channels_out, kernel_size=3, stride=stride, padding=1, dilation=1, bias=False)
        last_dim_out = unique_last_dim if unique_last_dim else channels_out
        self.conv3 = nn.Conv2d(channels_out, last_dim_out, kernel_size=1, stride=1, bias=False)
        self.output_act = output_act()

    def forward(self, x):
        out = x
        out = F.relu(self.conv1(out))
        out = F.relu(self.conv2(out))
        if self.residual:
            out = self.output_act(self.conv3(out) + x)
        else:
            out = self.output_act(self.conv3(out))
        return out