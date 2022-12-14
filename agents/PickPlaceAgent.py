from models.PickModel import PickModel
from models.PlaceModel import PlaceModel
import torch
from src.scripts.utils import get_affordance_map_from_formatted_input
import numpy as np
from models.CLIPWrapper import CLIPWrapper

class PickPlaceAgent:
    def __init__(self, num_rotations, lr, device):
        clip_model = CLIPWrapper()
        self.pick_model = PickModel(num_rotations=1, clip_model=clip_model).to(device)
        self.place_model = PlaceModel(num_rotations=num_rotations, clip_model=clip_model, crop_size=64).to(device)

        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(list(self.pick_model.parameters()) + list(self.place_model.parameters()), lr)
        self.num_rotations = num_rotations
        self.device = device
    
    # inp.keys() = dict_keys(['img', 'p0', 'p0_theta', 'p1', 'p1_theta', 'perturb_params', 'lang_goal'])
    def train_agent(self, inp):
        self.pick_model.train()
        self.place_model.train()

        p0 = np.array(inp['p0'])
        p0_rad = inp['p0_theta']
        p0_deg = np.rad2deg(p0_rad)

        p1 = np.array(inp['p1'])
        p1_rad = inp['p1_theta']
        p1_deg = np.rad2deg(p1_rad)
        
        pick_output_size = (1, 320, 160)
        place_output_size = (self.num_rotations, 320, 160)

        img_cuda = torch.Tensor(inp['img']).to(self.device)
        language_goal = inp['lang_goal']

        pick_demonstration = get_affordance_map_from_formatted_input(x=p0[0], y=p0[1], rotation_deg=p0_deg, output_size=pick_output_size)
        pick_affordances = self.pick_model(img_cuda, language_goal)
        pick_affordances = pick_affordances.view(pick_affordances.shape[0], -1)
        pick_demonstration = torch.unsqueeze(pick_demonstration, dim=0).to(self.device)
        pick_demonstration = pick_demonstration.view(pick_demonstration.shape[0], -1)
        pick_loss = self.loss_fn(pick_affordances, pick_demonstration)
        pick_location = np.unravel_index(torch.argmax(pick_affordances).item(), (320,160))
        pick_dist_err = np.linalg.norm(np.array(pick_location) - p0)
            
        place_demonstration = get_affordance_map_from_formatted_input(x=p1[0], y=p1[1], rotation_deg=p1_deg, output_size=place_output_size)
        place_affordances = self.place_model(img_cuda, language_goal, p0)
        place_affordances = place_affordances.view(place_affordances.shape[0], -1)
        place_demonstration = torch.unsqueeze(place_demonstration, dim=0).to(self.device)
        place_demonstration = place_demonstration.view(place_demonstration.shape[0], -1)
        place_loss = self.loss_fn(place_affordances, place_demonstration)
        place_location = np.unravel_index(torch.argmax(place_affordances).item(), (12,320,160))[1:]
        place_dist_err = np.linalg.norm(np.array(place_location) - p1)  

        loss = pick_loss + place_loss

        # propogate backwards
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {"pick_loss": pick_loss.item(), "place_loss": place_loss.item(),
                "pick_dist_error": pick_dist_err, "place_dist_error": place_dist_err}
    
    def eval_agent(self, inp):
        self.pick_model.eval()
        self.place_model.eval()
        with torch.no_grad():
            p0 = np.array(inp['p0'])
            p0_rad = inp['p0_theta']
            p0_deg = np.rad2deg(p0_rad)

            p1 = np.array(inp['p1'])
            p1_rad = inp['p1_theta']
            p1_deg = np.rad2deg(p1_rad)
            
            pick_output_size = (1, 320, 160)
            place_output_size = (self.num_rotations, 320, 160)

            img_cuda = torch.Tensor(inp['img']).to(self.device)
            language_goal = inp['lang_goal']

            pick_demonstration = get_affordance_map_from_formatted_input(x=p0[0], y=p0[1], rotation_deg=p0_deg, output_size=pick_output_size)
            pick_affordances = self.pick_model(img_cuda, language_goal)
            pick_affordances = pick_affordances.view(pick_affordances.shape[0], -1)
            pick_demonstration = torch.unsqueeze(pick_demonstration, dim=0).to(self.device)
            pick_demonstration = pick_demonstration.view(pick_demonstration.shape[0], -1)
            pick_loss = self.loss_fn(pick_affordances, pick_demonstration)
            pick_location = np.unravel_index(torch.argmax(pick_affordances).item(), (320,160))
            pick_dist_err = np.linalg.norm(np.array(pick_location) - p0)
                
            place_demonstration = get_affordance_map_from_formatted_input(x=p1[0], y=p1[1], rotation_deg=p1_deg, output_size=place_output_size)
            place_affordances = self.place_model(img_cuda, language_goal, p0)
            place_affordances = place_affordances.view(place_affordances.shape[0], -1)
            place_demonstration = torch.unsqueeze(place_demonstration, dim=0).to(self.device)
            place_demonstration = place_demonstration.view(place_demonstration.shape[0], -1)
            place_loss = self.loss_fn(place_affordances, place_demonstration)
            place_location = np.unravel_index(torch.argmax(place_affordances).item(), (12,320,160))[1:]
            place_dist_err = np.linalg.norm(np.array(place_location) - p1)  
                               
        return {"pick_loss": pick_loss.item(), "place_loss": place_loss.item(),
                "pick_dist_error": pick_dist_err, "place_dist_error": place_dist_err}

