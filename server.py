from flask import Flask, request, jsonify
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
import pickle

from nnutils.vposer_predictor import VPoserPredictor
from nnutils.vposer import VPoser

app = Flask(__name__)

device = torch.device("cpu")
SKELETON = "data/skeleton.pt"
VPOSER   = "model/vposer.pt"
model = VPoserPredictor(skeleton_path=SKELETON, vposer_path=VPOSER)
model.to(device)
optimizer = optim.Adam([model.pose_embedding, model.global_rotation, model.global_trans], lr=2e-1)

def predict_pose(joint_pos, joint_mask, iteration=100):
    model.pose_embedding.requires_grad = True
    model.global_rotation.requires_grad = True
    model.global_trans.requires_grad = True

    for it in range(0, iteration):
        optimizer.zero_grad()
        joint_pos_predicted = model.forward()
        loss = F.mse_loss(joint_pos_predicted*joint_mask, joint_pos*joint_mask) + 1e-3 * model.pose_embedding.pow(2).sum()
        loss.backward()
        optimizer.step()
        # print("Iteration {}, loss at {}".format(it, loss.item()))

    pose, trans = model.get_pose()
    return VPoser.aa2matrot(pose.view(1,1,-1,9)), trans

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
        pose, trans = predict_pose(joint_pos.to(device), joint_mask.to(device))
        return jsonify({'pose': pose.view(-1,9).detach().cpu().numpy().tolist(), "trans": trans.detach().cpu().numpy().tolist()})

if __name__ == '__main__':
    app.run(port=1028)