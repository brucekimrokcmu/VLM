from models.PickModel import PickModel
from models.PlaceModel import PlaceModel
import torch
from src.scripts.utils import get_affordance_map_from_formatted_input
import numpy as np
from models.CLIPWrapper import CLIPWrapper
from cliport.utils import utils

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
        

    def act(self, img, lang_goal):
        inp = {'inp_img': img, 'lang_goal': lang_goal}

        self.pick_model.eval()
        self.place_model.eval()
        with torch.no_grad():
            img_cuda = torch.Tensor(inp['inp_img']).to(self.device)
            language_goal = inp['lang_goal']

            pick_affordances = self.pick_model(img_cuda, language_goal)
            self.pick_model(img_cuda, language_goal)
            pick_affordances = pick_affordances.view(pick_affordances.shape[0], -1)
            pick_preds = torch.nn.functional.softmax(pick_affordances, dim=1)
            pick_preds = pick_preds.cpu()
            pick_preds = pick_preds.view(320,160)
            p0_pix = np.unravel_index(torch.argmax(pick_preds).numpy(), (320,160))
            p0_theta = 0

            place_affordances = self.place_model(img_cuda, language_goal, p0_pix)
            place_affordances = place_affordances.view(place_affordances.shape[0], -1)
            place_preds = torch.nn.functional.softmax(place_affordances, dim=1)
            place_preds = place_preds.cpu()
            place_preds = place_preds.view(12, 320,160)
            p1 = np.unravel_index(torch.argmax(place_preds).numpy(), (self.num_rotations, 320,160))
            p1_pix = p1[1:3]
            p1_theta = p1_pix[0] * 2 * np.pi / self.num_rotations
        
            # Pixels to end effector poses.
            bounds = np.array([[0.25, 0.75], [-0.5, 0.5], [0, 0.28]])
            pix_size = 0.003125

            hmap = img[:, :, 3]
            p0_xyz = utils.pix_to_xyz(p0_pix, hmap, bounds, pix_size)
            p1_xyz = utils.pix_to_xyz(p1_pix, hmap, bounds, pix_size)
            p0_xyzw = utils.eulerXYZ_to_quatXYZW((0, 0, -p0_theta))
            p1_xyzw = utils.eulerXYZ_to_quatXYZW((0, 0, -p1_theta))

            act = {
                'pose0': (np.asarray(p0_xyz), np.asarray(p0_xyzw)),
                'pose1': (np.asarray(p1_xyz), np.asarray(p1_xyzw)),
                'pick': [p0_pix[0], p0_pix[1], p0_theta],
                'place': [p1_pix[0], p1_pix[1], p1_theta],
            }

        return act, (pick_affordances, place_affordances)

