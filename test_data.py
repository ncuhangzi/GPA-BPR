import torch
import time

textural_idx_dict = "./feat/visualfeatures"
text = torch.load(textural_idx_dict, map_location= lambda a,b:a.cpu())

print(text)
