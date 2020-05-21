import torch
from torch import nn
from .vposer import VPoser
from .skeleton import Skeleton

class VPoserPredictor(nn.Module):

    def __init__(self, skeleton_path, vposer_path):
        super(VPoserPredictor, self).__init__()

        self.ske = Skeleton(skeleton_path=skeleton_path)
        self.vposer = VPoser(512, 32, (1,31,3))
        self.vposer.load_state_dict(torch.load(vposer_path))
        self.vposer.eval()
        for p in self.vposer.parameters():
            p.requires_grad = False

        pose_embedding = torch.zeros(1, 32).cuda()
        self.register_buffer('pose_embedding', pose_embedding.type(torch.float32))

        global_rotation = torch.zeros(1, 1, 3).cuda()
        self.register_buffer('global_rotation', global_rotation.type(torch.float32))

        global_trans = torch.zeros(1, 1, 3).cuda()
        self.register_buffer('global_trans', global_trans.type(torch.float32))

    def forward(self):
        pose, trans = self.get_pose()
        joints = self.ske(pose, trans)
        return joints

    def get_pose(self):
        body_pose = self.vposer.decode(self.pose_embedding, output_type='aa').view(1, 31, 3)
        pose = torch.cat([self.global_rotation, body_pose], dim=1)
        return pose, self.global_trans


