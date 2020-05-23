# -*- coding: utf-8 -*-
#
# Author: maajor <info@ma-yidong.com>
# Date : 2020-05-23
# 
# a debug util to sample from training data and for visualization in blender purpose

import time
import pickle
import torch
import numpy as np

from .skeleton import Skeleton
from .dataloader import AnimationDS
from .vposer import VPoser


ds_train = AnimationDS("data/train/pose_train.pt")
device = torch.device('cpu')

pose = ds_train[3235]['pose_aa'].to(device).view(1,-1, 3)
print(pose.shape)
trans = torch.from_numpy(np.zeros(3)).type(torch.float32).to(device)

model = Skeleton(skeleton_path='data/skeleton.pt')
model.to(device)
j = model(pose.view(1,-1, 3), trans.view(1, 3))

theta = torch.sqrt(torch.sum((pose)**2, dim=2)).view(1, -1, 1)
r_hat = pose / theta
aa = torch.zeros((31,4))
aa[:,0:1] = theta.view(-1,1)
aa[:,1:] = r_hat.view(-1,3)

result={}
result["pose"] = aa.view(-1,4).detach().cpu().numpy().tolist()
result["trans"] = trans.detach().cpu().numpy().tolist()
result["joints"] = j.detach().cpu().numpy().tolist()
with open("pose.pkl", "wb") as f:
    pickle.dump(result, f)