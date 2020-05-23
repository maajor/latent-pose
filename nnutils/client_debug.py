# -*- coding: utf-8 -*-
#
# Author: maajor <info@ma-yidong.com>
# Date : 2020-05-23
# 
# a debug util to query pose from server

import requests
import pickle
import torch

param_dict = {}
param_dict["joint_pos"] =  [0, 0, 0,-2, 6, 6, 1, -1, 7]
param_dict["joint_id"] = [0, 20, 16] # left hand and head
resp = requests.get("http://127.0.0.1:1028/predict",
                     params=param_dict)
result = resp.json()
print(result)
with open("pose.pkl", "wb") as f:
    pickle.dump(result, f)