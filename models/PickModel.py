import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.functional as TF
from torchvision import datasets, transforms

from models.SpatialSemanticStream import SpatialSemanticStream

class PickModel(nn.Module):
  def __init__(self, num_rotations, batchnorm = False):
    super().__init__()
    self.num_rotations = num_rotations
    self.model = SpatialSemanticStream(channels_in=6, pick=True, batchnorm=batchnorm) 
    self.tensor_to_PIL = transforms.ToPILImage()   

  def forward(self, rgb_ddd_img, language_command):
    # Rotate input multiple times and run forward for each one
    out_all = []
    for i in range(self.num_rotations):
      angle = i * 360/self.num_rotations
      
      # print(type(rgb_ddd_img[0]))
      swapped_img = torch.unsqueeze(torch.swapaxes(torch.swapaxes(rgb_ddd_img[0], 1, 2), 0, 1), dim=0) 
      # print(swapped_img.shape)
      rotated_img = TF.rotate(swapped_img, angle, torchvision.transforms.InterpolationMode.BILINEAR)
      # Un-rotate output
      out = self.model(rotated_img, language_command[:1])
      out = TF.rotate(out, -angle, torchvision.transforms.InterpolationMode.BILINEAR)
      out_all.append(out)
    out_all = torch.cat(out_all, dim=1)
    return out_all
