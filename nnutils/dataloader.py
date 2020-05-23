# -*- coding: utf-8 -*-
#
# Author: maajor <info@ma-yidong.com>
# Date : 2020-05-23

import glob, os

import torch
from torch.utils.data import Dataset

import numpy as np
from .vposer import VPoser

class AnimationDS(Dataset):
    def __init__(self, dataset_dir):
        
        self.ds = torch.load(dataset_dir)

    def __len__(self):
        return self.ds.shape[0]

    def __getitem__(self, idx):
        return self.fetch_data(idx)

    def fetch_data(self, idx):
        pose = self.ds[idx]
        data = {}
        data['pose_aa'] = pose.view(1,-1,3)
        data['pose_matrot'] = VPoser.aa2matrot(data['pose_aa'][np.newaxis]).view(1,-1,9)
        return data