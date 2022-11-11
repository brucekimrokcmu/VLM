# TODO UPDATE FROM KEVIN
import pickle
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import torchvision
import clip
device = "cuda"

class ClipPort6D(nn.Module):
    def __init__(self, cfg, device="cuda"):
        super().__init__()
        self.clip_model, self.preprocess = clip.load("RN50")
        modified_resnet = list(self.clip_model.children())[0]
        self.modified_resnet = torch.nn.Sequential(*list(modified_resnet.children())[:-1])
        self.device = device

        for param in self.clip_model.parameters():
            param.requires_grad = False

        # TODO set requires grad to false
    
    def forward(self, rgb_image):
        head = self.modified_resnet[0:10](rgb_image.half())
        layer1 = self.modified_resnet[10](head)
        layer2 = self.modified_resnet[11](layer1)
        layer3 = self.modified_resnet[12](layer2)
        layer4 = self.modified_resnet[13](layer3)
        return layer1.float(), layer2.float(), layer3.float(), layer4.float()
    
    def embed_sentence(self, language_commands):
        tokens = torch.cat([clip.tokenize(c for c in language_commands)]).to(device)
        # tokens = clip.tokenize(language_command).to(self.device)
        sentence_embedding = self.clip_model.encode_text(tokens)
        return sentence_embedding.float()