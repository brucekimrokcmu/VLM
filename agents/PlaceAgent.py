from models.PlaceModel import PlaceModel
import torch
from src.scripts.utils import get_affordance_map_from_formatted_input
import numpy as np

class PlaceAgent:
    def __init__(self, num_rotations, lr, clip_model):
        # Place model, alson with loss functions
        self.model = PlaceModel(num_rotations=num_rotations, clip_model=clip_model, crop_size=64)
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr)
        self.scheduler = None # torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer)
        self.num_rotations = num_rotations
    
    # inp.keys() = dict_keys(['img', 'p0', 'p0_theta', 'p1', 'p1_theta', 'perturb_params', 'lang_goal'])
    def train_agent(self, inp):
        self.model.train()

        p0 = inp['p0']
        p1 = inp['p1']
        p1_rad = inp['p1_theta']
        p1_deg = np.rad2deg(p1_rad)
        output_size = (self.num_rotations, 320, 160)

        place_demonstration = get_affordance_map_from_formatted_input(x=p1[0], y=p1[1], rotation_deg=p1_deg, output_size=output_size)
        device = 'cuda' # TODO: Not do this
        img_cuda = torch.Tensor(inp['img']).to(device)
        language_cuda = inp['lang_goal']#.to(device)
        affordances = self.model(img_cuda, language_cuda, p0)
        affordances = affordances.view(affordances.shape[0], -1)
        place_demonstration = torch.unsqueeze(place_demonstration, dim=0).to(device)
        place_demonstration = place_demonstration.view(place_demonstration.shape[0], -1)
        loss = self.loss_fn(affordances, place_demonstration)

        # propogate backwards
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # self.scheduler.step(loss)
        return loss.item()
    
    def eval_agent(self, inp):
        self.model.eval()
        with torch.no_grad():
            p0 = inp['p0']
            p1 = inp['p1']
            p1_rad = inp['p1_theta']
            p1_deg = np.rad2deg(p1_rad)
            output_size = (self.num_rotations, 320, 160)

            place_demonstration = get_affordance_map_from_formatted_input(x=p1[0], y=p1[1], rotation_deg=p1_deg, output_size=output_size)
            device = 'cuda' # TODO: Not do this
            img_cuda = torch.Tensor(inp['img']).to(device)
            language_cuda = inp['lang_goal']#.to(device)
            affordances = self.model(img_cuda, language_cuda, p0)
            affordances = affordances.view(affordances.shape[0], -1)
            place_demonstration = torch.unsqueeze(place_demonstration, dim=0).to(device)
            place_demonstration = place_demonstration.view(place_demonstration.shape[0], -1)
            loss = self.loss_fn(affordances, place_demonstration)
            
        return loss.item()
