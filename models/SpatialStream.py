import torch
import torch.nn as nn
import torch.nn.functional as F
from cliport.utils import utils

from models.blocks.ConvBlock import ConvBlock

class SpatialStream(nn.Module):
    def __init__(self, channels_in, channels_out):
      super().__init__()
      self.channels_in = channels_in
      self.channels_out = channels_out
      self._make_layers()
    
    def _make_layers(self):
      # 1x conv
      self.conv = nn.Sequential(
          nn.Conv2d(self.channels_in, 64, kernel_size=3, stride=1, padding=1),
          nn.ReLU(True)
      )
      # Encoder: 6x conv & identity (downsample)
      self.encoder1 = nn.Sequential(
          ConvBlock(64, 64, 1),
          ConvBlock(64, 64, 1, False, True)
      )
      self.encoder2 = nn.Sequential(
          ConvBlock(64, 128, 2),
          ConvBlock(128, 128, 1, False, True)
      )
      self.encoder3 = nn.Sequential(
          ConvBlock(128, 256, 2),
          ConvBlock(256, 256, 1, False, True)
      )
      self.encoder4 = nn.Sequential(
          ConvBlock(256, 512, 2),
          ConvBlock(512, 512, 1, False, True)
      )
      self.encoder5 = nn.Sequential(
          ConvBlock(512, 1024, 2),
          ConvBlock(1024, 1024, 1, False, True)
      )

      self.decoder1 = nn.Sequential(
          ConvBlock(1024, 512, 1),
          ConvBlock(512, 512, 1, False, True),
          nn.UpsamplingBilinear2d(scale_factor=2)
      )
      self.decoder2 = nn.Sequential(
          ConvBlock(512, 256, 1),
          ConvBlock(256, 256, 1, False, True),
          nn.UpsamplingBilinear2d(scale_factor=2)
      )
      self.decoder3 = nn.Sequential(
          ConvBlock(256, 128, 1),
          ConvBlock(128, 128, 1, False, True),
          nn.UpsamplingBilinear2d(scale_factor=2)
      )
      self.decoder4 = nn.Sequential(
          ConvBlock(128, 64, 1),
          ConvBlock(64, 64, 1, False, True),
          nn.UpsamplingBilinear2d(scale_factor=2)
      )
      self.decoder5 = nn.Sequential(
          ConvBlock(64, 32, 1),
          ConvBlock(32, 32, 1, False, True),
          nn.UpsamplingBilinear2d(scale_factor=2)
      )
      # 1x conv & identity & downsample
      self.down = nn.Sequential(
          ConvBlock(32, 16, 1, self.channels_out, output_act=nn.Identity),
          ConvBlock(self.channels_out, 16, 1, self.channels_out, True, output_act=nn.Identity),
      )

    def forward(self, rgb_ddd_img):
      # Conv
      rgb_ddd_img = utils.preprocess(rgb_ddd_img, dist='transporter')
      out = self.conv(rgb_ddd_img)
      #Encoder
      for encode in [self.encoder1, self.encoder2, self.encoder3, self.encoder4, self.encoder5]:
        out = encode(out)
      #Decoder
      lateral_out = [out]
      for decode in [self.decoder1, self.decoder2, self.decoder3, self.decoder4, self.decoder5]:
        out = decode(out)
        lateral_out.append(out)
      # Downsample
      out = self.down(out)
      out = F.interpolate(out, size=(rgb_ddd_img.shape[-2], rgb_ddd_img.shape[-1]), mode='bilinear')
      # Return output and lateral output
      return out, lateral_out
