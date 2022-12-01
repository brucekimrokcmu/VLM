import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleModel(nn.Module):
  def __init__(self, num_rotations, batchnorm = False):
    super().__init__()
    self.num_rotations = num_rotations
    self.input_size = 224 * 224 * 4 // 64 # RGBD
    self.output_size = 224 * 224 * num_rotations # img x rotations
    self.model = self.__make_model__()

  def __make_model__(self):
    layer = nn.Sequential(
        nn.Linear(self.input_size, self.output_size),
        nn.ReLU()
    )
    return layer

  def forward(self, rgb_ddd_img, language_command):
    # Downsample, flatten, forward, shape output, return
    # rgb_ddd_img = [2, dim, dim, RGBDDD]
    output = rgb_ddd_img[0,::8,::8,:4]
    output = torch.flatten(output)
    output = self.model(output)
    output = torch.reshape(output, (1, self.num_rotations, 224, 224))
    return output
