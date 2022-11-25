from models.PickModel import PickModel
import torch
from src.scripts.utils import convert_angle_to_channel, get_affordance_map_from_formatted_input
class PickAgent:
    def __init__(self, num_rotations, lr):
        # Pick model, alson with loss functions
        self.model = PickModel(num_rotations=num_rotations, batchnorm = False)
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr)
        # self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer)
        self.num_rotations = num_rotations
    
    # inp = {'img':img, 'lang_goal': language_instructions,
    #     'p0':p0, 'p0_z':p0_z, 'p0_rotation':p0_rotation,
    #     'p1':p1, 'p1_z':p1_z, 'p1_rotation':p1_rotation}
    def train_agent(self, inp):
        self.model.train()
        
        loss_dict = {}
        total_loss = []

        p0 = inp['p0']
        yaw_deg = inp['p0_rotation'][0]
        output_size = (self.num_rotations, 224, 224)

        print(inp.keys())
        pick_demonstration = get_affordance_map_from_formatted_input(x=p0[0,0], y=p0[0,1], rotation_deg=yaw_deg, output_size=output_size)
        device = 'cuda' # TODO: Not do this
        img_cuda = torch.Tensor(inp['img']).to(device)
        language_cuda = inp['lang_goal']#.to(device)
        affordances = self.model(img_cuda, language_cuda)
        loss = self.loss_fn(affordances, pick_demonstration)
        # output_size = inp['img'].shape
            
        # propogate backwards
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # self.scheduler.step(loss)
    
    def eval(self, inp_dict):
        pass

