"""
MIT License

Copyright (c) 2022 UCSC ERIC Lab

The utilities below used substantial portions of https://github.com/eric-ai-lab/VLMbench.git
"""
import datetime
import numpy as np
from scipy.spatial.transform import Rotation as R

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

def FormatInput(batch_data):
    if len(batch_data)==0:
        return None 

    bounds, pixel_size = batch_data[0]['bounds'], batch_data[0]['pixel_size']
    img, language_instructions = [], []
    attention_points, target_points = [], []
    
    for data in batch_data:
            img.append(data['img'])
            language_instructions += data['language']
            attention_points.append(data['attention_points'])
            target_points.append(data['target_points'])

    img =  np.concatenate(img, axis=0)
    attention_points =  np.concatenate(attention_points, axis=0)
    target_points =  np.concatenate(target_points, axis=0)

    if (attention_points[:, :2]>bounds[:2,1]).any() or (attention_points[:, :2]<bounds[:2,0]).any() \
            or (target_points[:, :2]>bounds[:2,1]).any() or (target_points[:, :2]<bounds[:2,0]).any():
            return None

    p0 = np.int16((attention_points[:, :2]-bounds[:2,0])/pixel_size)
    p0_z = attention_points[:, 2:3]-bounds[2,0]
    p0_rotation = R.from_quat(attention_points[:, 3:])
    p1 = np.int16((target_points[:, :2]-bounds[:2,0])/pixel_size)
    p1_z = target_points[:, 2:3]-bounds[2,0]
    p1_rotation = R.from_quat(target_points[:, 3:])
    p0 = p0[:,::-1]
    p1 = p1[:,::-1]
    p0_rotation = p0_rotation.as_euler('zyx', degrees=True)
    p1_rotation = p1_rotation.as_euler('zyx', degrees=True)

    inp = {'img':img, 'lang_goal': language_instructions,
        'p0':p0, 'p0_z':p0_z, 'p0_rotation':p0_rotation,
        'p1':p1, 'p1_z':p1_z, 'p1_rotation':p1_rotation}

    return inp

def convert_angle_to_channel(angle_deg, num_rotations):
    i = angle_deg * num_rotations // 360
    return i

def get_affordance_map_from_formatted_input(x, y, rotation_deg, output_size):
    num_rotations = output_size.shape[0]
    rotation_channel = convert_angle_to_channel(rotation_deg, num_rotations)
    affordance_map = torch.zeros(output_size)
    affordance_map[rotation_channel, x, y] = 1
    return affordance_map
