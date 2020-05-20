import torch
import numpy as np
from .skeleton import Skeleton
from .vposer import VPoser
import time
import pickle

device = torch.device("cuda")

vposer = VPoser(512,32,(1, 31 ,3))
vposer.load_state_dict(torch.load("model/vposer.pt", map_location=device))
vposer.to(device)
vposer.eval()

sampled_pose = vposer.sample_poses(1, seed=int((time.time()%1000)*1000)).view(-1,3)

root_rot = torch.tensor([0,0,0]).type(torch.float32).view(1,3).to(device)

pose = torch.cat([root_rot,sampled_pose], dim=0).to(device)
trans = torch.from_numpy(np.zeros(3)).type(torch.float32).to(device)

model = Skeleton(skeleton_path="data/skeleton.pt")
model.to(device)
j = model(pose.view(1,-1, 3), trans.view(1, 3))
j = j.detach().cpu().numpy()

with open("joints.pkl", "wb") as f:
    pickle.dump(j, f)