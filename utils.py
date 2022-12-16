"""
MIT License

Copyright (c) 2022 UCSC ERIC Lab

The utilities below used substantial portions of https://github.com/eric-ai-lab/VLMbench.git
"""
import torch
import datetime

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def sec_to_str(delta):
    t = datetime.timedelta(seconds=delta)
    s = str(t)
    return s.split(".")[0] + "s"

def convert_angle_to_channel(angle_deg, num_rotations):
    i = int(angle_deg * num_rotations // 360)
    return i

def get_affordance_map_from_formatted_input(x, y, rotation_deg, output_size):
    num_rotations = output_size[0]
    rotation_channel = convert_angle_to_channel(rotation_deg, num_rotations)
    affordance_map = torch.zeros(output_size)
    affordance_map[rotation_channel, x, y] = 1
    return affordance_map
