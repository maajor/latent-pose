# -*- coding: utf-8 -*-
#
# Author: maajor <info@ma-yidong.com>
# Date : 2020-05-23
# 
# A simple server to predict IK poses

from flask import Flask, request, jsonify
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
import pickle
import time

from nnutils.vposer_predictor import VPoserPredictor
from nnutils.vposer import VPoser
from nnutils.geoutils import *

app = Flask(__name__)

device = torch.device("cpu")

SKELETON = "data/skeleton.pt"
VPOSER   = "model/vposer.pt"
model = VPoserPredictor(skeleton_path=SKELETON, vposer_path=VPOSER)
model.to(device)
optimizer = optim.Adam([model.pose_embedding, model.global_trans], lr=3e-1)

def predict_pose(joint_pos, joint_mask, iteration=100):
    start = time.time()
    model.pose_embedding.requires_grad = True
    model.global_trans.requires_grad = True

    for it in range(0, iteration):
        optimizer.zero_grad()
        joint_pos_predicted = model.forward()
        loss = F.mse_loss(joint_pos_predicted*joint_mask, joint_pos*joint_mask) + 1e-3 * model.pose_embedding.pow(2).sum()
        loss.backward()
        optimizer.step()
        # print("Iteration {}, loss at {}".format(it, loss.item()))

    pose, trans = model.get_pose()

    # pose decode to blender's axis-angle float4 format
    theta = torch.sqrt(torch.sum((pose)**2, dim=2)).view(1, -1, 1)
    r_hat = pose / theta
    aa = torch.zeros((31,4))
    aa[:,0:1] = theta.view(-1,1)
    aa[:,1:] = r_hat.view(-1,3)

    done = time.time()
    elapsed = done - start
    print("Time used "+ str(elapsed))
    return aa, trans.view(3), joint_pos_predicted

@app.route('/predict', methods=['GET'])
def predict():
    if request.method == 'GET':
        j_pos = request.args.getlist("joint_pos")
        j_id  = request.args.getlist("joint_id")
        joint_pos = torch.zeros((1,31,3)).type(torch.float32)
        joint_mask = torch.zeros((1,31,3)).type(torch.float32)
        for i, jid in enumerate(j_id):
            jid = int(jid)
            joint_mask[0,jid,:] = 1
            joint_pos[0,jid,0] = float(j_pos[i*3])
            joint_pos[0,jid,1] = float(j_pos[i*3+1])
            joint_pos[0,jid,2] = float(j_pos[i*3+2])
        pose, trans, j = predict_pose(joint_pos.to(device), joint_mask.to(device))
        return jsonify({
            'pose': pose.detach().cpu().numpy().tolist(), 
            "trans": trans.detach().cpu().numpy().tolist()})

if __name__ == '__main__':
    app.run(port=1028)