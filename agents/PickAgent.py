from models.PickModel import PickModel
import torch
from src.utils import convert_angle_to_channel
class PickAgent:
    def __init__(self, num_rotations, lr):
        # Pick model, alson with loss functions
        self.model = PickModel(num_rotations=num_rotations, batchnorm = False)
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    
    # inp = {'img':img, 'lang_goal': language_instructions,
    #     'p0':p0, 'p0_z':p0_z, 'p0_rotation':p0_rotation,
    #     'p1':p1, 'p1_z':p1_z, 'p1_rotation':p1_rotation}
    def train(self, inp_dict):
        self.model.train()
        
        loss_dict = {}
        total_loss = []

        p0 = inp['p0']
        yaw_deg = inp['p0_rotation'][0]
        

        pick_demonstration = get_affordance_map_from_formatted_input(x=p0[0], y=p0[1], rotation_deg=yaw_deg, output_size)
        affordances = self.model(inp['img'], inp['lang_goal'])
        loss = self.loss_fn(affordances, pick_demonstration)
        # output_size = inp['img'].shape
            
        # propogate backwards
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(loss)
    
    def eval(self, inp_dict):
        pass

