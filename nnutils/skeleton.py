import numpy as np
import pickle
import torch
from torch.nn import Module
import os


class Skeleton(Module):
  def __init__(self, skeleton_path='data/skeleton.pkl'):
    
    super(Skeleton, self).__init__()
    with open(skeleton_path, 'rb') as f:
      params = pickle.load(f)
    # joints position, n_joint * 3
    #self.J = torch.from_numpy(params['J']).type(torch.float32)
    self.register_buffer('J', torch.from_numpy(params['J']).type(torch.float32))
    self.J.requires_grad = False
    self.J_num = self.J.size()[0]
    # parent joint id to child joint id mapping, 2 * n_joints
    self.kintree_table = params['kintree_table']

  @staticmethod
  def rodrigues(r):
    """
    Rodrigues' rotation formula that turns axis-angle tensor into rotation
    matrix in a batch-ed manner.

    Parameter:
    ----------
    r: Axis-angle rotation tensor of shape [batch_size, 1, 3].

    Return:
    -------
    Rotation matrix of shape [batch_size, 3, 3].

    """
    #r = r.to(self.device)
    eps = r.clone().normal_(std=1e-8)
    # why not work in pytorch1.4.0
    #theta = torch.norm(r + eps, dim=(1, 2), keepdim=True)  # dim cannot be tuple
    theta = torch.sqrt(torch.sum((r + eps)**2, dim=2)).view(-1,1,1)
    theta_dim = theta.shape[0]
    r_hat = r / theta
    cos = torch.cos(theta)
    z_stick = torch.zeros(theta_dim, dtype=torch.float32).to(r.device)
    m = torch.stack(
      (z_stick, -r_hat[:, 0, 2], r_hat[:, 0, 1], r_hat[:, 0, 2], z_stick,
       -r_hat[:, 0, 0], -r_hat[:, 0, 1], r_hat[:, 0, 0], z_stick), dim=1)
    m = torch.reshape(m, (-1, 3, 3))
    i_cube = (torch.eye(3, dtype=torch.float32).unsqueeze(dim=0) \
             + torch.zeros((theta_dim, 3, 3), dtype=torch.float32)).to(r.device)
    A = r_hat.permute(0, 2, 1)
    dot = torch.matmul(A, r_hat)
    R = cos * i_cube + (1 - cos) * dot + torch.sin(theta) * m
    return R

  @staticmethod
  def with_zeros(x):
    """
    Append a [0, 0, 0, 1] tensor to a [3, 4] tensor.

    Parameter:
    ---------
    x: Tensor to be appended.

    Return:
    ------
    Tensor after appending of shape [4,4]

    """
    ones = torch.tensor(
      [[[0.0, 0.0, 0.0, 1.0]]], dtype=torch.float32
    ).expand(x.shape[0],-1,-1).to(x.device)
    ret = torch.cat((x, ones), dim=1)
    return ret

  @staticmethod
  def pack(x):
    """
    Append zero tensors of shape [4, 3] to a batch of [4, 1] shape tensor.

    Parameter:
    ----------
    x: A tensor of shape [batch_size, 4, 1]

    Return:
    ------
    A tensor of shape [batch_size, 4, 4] after appending.

    """
    zeros43 = torch.zeros(
      (x.shape[0], x.shape[1], 4, 3), dtype=torch.float32).to(x.device)
    ret = torch.cat((zeros43, x), dim=3)
    return ret

  def write_obj(self, file_name):
    with open(file_name, 'wb') as fp:
      pickle.dump(self.J_posed.detach().cpu().numpy(), fp)

  def forward(self, pose, trans):
    """
          Construct a compute graph that takes in parameters and outputs a tensor as
          model vertices. Face indices are also returned as a numpy ndarray.

          Prameters:
          ---------
          pose: Also known as 'theta', a [n_pose_joint,3] tensor indicating child joint rotation
          relative to parent joint. For root joint it's global orientation.
          Represented in a axis-angle format.

          trans: Global translation tensor of shape [3].

          Return:
          ------
          A tensor for vertices, and a numpy ndarray as face indices.

    """
    id_to_col = {i: self.kintree_table[0, i]
                 for i in range(self.kintree_table.shape[1])}
    parent = {
      self.kintree_table[1, i]: self.kintree_table[0, i]
      for i in range(1, self.kintree_table.shape[1])
    }

    J = self.J

    batch_num = pose.shape[0]

    J = self.J.view(1,-1,3).repeat(batch_num,1,1)
    R_cube_big = self.rodrigues(pose.view(-1, 1, 3)).reshape(batch_num, -1, 3, 3)


    # transform matrix of each joints
    results = []
    results.append(
      self.with_zeros(torch.cat((R_cube_big[:, 0], torch.reshape(J[:, 0, :], (-1, 3, 1))), dim=2))
    )
    for i in range(1, self.kintree_table.shape[1]):
      results.append(
        torch.matmul(
          results[parent[i]],
          self.with_zeros(
            torch.cat(
              (R_cube_big[:, i], torch.reshape(J[:, i, :] - J[:, parent[i], :], (-1, 3, 1))),
              dim=2
            )
          )
        )
      )
    
    stacked = torch.stack(results, dim=1)

    # posed joint position
    self.J_posed = stacked[:,:,:3,3]

    self.J_posed = self.J_posed + torch.reshape(trans, (batch_num, 1, 3))

    return self.J_posed


def main():
  pose_size = 31*3

  device = torch.device("cpu")

  pose = torch.from_numpy((np.random.rand(pose_size) - 0.5) * 0.5)\
          .type(torch.float32).to(device)

  trans = torch.from_numpy(np.zeros(3)).type(torch.float32).to(device)

  model = Skeleton(model_path='data/skeleton.pkl')
  model.to(device)
  j = model(pose.view(1,-1, 3), trans.view(1, 3))
  model.write_obj("joints.pkl")

if __name__ == '__main__':
  main()
