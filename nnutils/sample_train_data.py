import time
import pickle
import torch
import numpy as np

from .skeleton import Skeleton
from .dataloader import AnimationDS
from .vposer import VPoser


ds_train = AnimationDS("data/train/pose_train.pt")
device = torch.device('cpu')

pose = ds_train[185]['pose_aa'].to(device)
trans = torch.from_numpy(np.zeros(3)).type(torch.float32).to(device)

model = Skeleton(skeleton_path='data/skeleton.pt')
model.to(device)
j = model(pose.view(1,-1, 3), trans.view(1, 3))
model.write_obj("joints.pkl")