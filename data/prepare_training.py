# -*- coding: utf-8 -*-
#
# Author: maajor <info@ma-yidong.com>
# Date : 2020-05-23
# 
# Util to convert data extracted from blender to VPoser's training data

import os
import pickle
import torch
import tqdm
import numpy as np
from nnutils.vposer import VPoser

extracted_anim_path = "data/extracted/"
extracted_anim_files = []
for r, d, f in os.walk(extracted_anim_path):
    extracted_anim_files.extend(["{}/{}".format(r,file) for file in f if file.endswith(".pkl")])
extracted_anim_files.sort()

loop = tqdm.tqdm(range(0, len(extracted_anim_files)))
poses = []
for i in loop:
    anim_file = extracted_anim_files[i]
    with open(anim_file, "rb") as f:
        anim_data = pickle.load(f)
        anim_data = torch.tensor(anim_data, dtype=torch.float32)
        anim_data = anim_data.reshape((anim_data.shape[0],1,anim_data.shape[1],9))
        aa = VPoser.matrot2aa(anim_data)
        poses.append(aa)
poses = torch.cat(poses, dim=0).squeeze(dim=1).numpy()

np.random.shuffle(poses)

lens = int(poses.shape[0])
pose_train = poses[:int(0.7*lens),:,:]
pose_val = poses[int(0.7*lens):int(0.9*lens),:,:]
pose_test = poses[int(0.9*lens):,:,:]
torch.save(torch.from_numpy(pose_train), "data/train/pose_train.pt")
torch.save(torch.from_numpy(pose_val), "data/train/pose_val.pt")
torch.save(torch.from_numpy(pose_test), "data/train/pose_test.pt")

#blender --background -P data/cmu-mocap/import_bvh_and_convert.py  --bvh data/cmu-mocap/01/01_02.bvh