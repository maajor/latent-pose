__all__ = []

import os, shutil
from datetime import datetime

import numpy as np

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
import tqdm

from .vposer import VPoser
from .skeleton import Skeleton
from .geoutils import *

class VPoserTrainer:

    def __init__(self, work_dir, skeleton_path):
        from .dataloader import AnimationDS

        self.batch_size = 10000

        self.pt_dtype = torch.float32

        self.comp_device = torch.device("cuda")

        ds_train = AnimationDS(work_dir+"_train.pt")
        self.ds_train = DataLoader(ds_train, batch_size=self.batch_size, shuffle=True, drop_last=True)

        ds_val = AnimationDS(work_dir+"_val.pt")
        self.ds_val = DataLoader(ds_val, batch_size=self.batch_size, shuffle=True, drop_last=True)

        ds_test = AnimationDS(work_dir+"_test.pt")
        self.ds_test = DataLoader(ds_test, batch_size=self.batch_size, shuffle=True, drop_last=True)

        print('Train dataset size %.2f M' % (len(self.ds_train.dataset)*1e-6))
        print('Validation dataset size %d' % len(self.ds_val.dataset))
        print('Test dataset size %d' % len(self.ds_test.dataset))

        data_shape = list(ds_val[0]['pose_aa'].shape)
        self.latentD = 32
        self.vposer_model = VPoser(num_neurons=512, latentD=self.latentD, data_shape=data_shape,
                                   use_cont_repr=True)

        self.vposer_model.to(self.comp_device)

        varlist = [var[1] for var in self.vposer_model.named_parameters()]

        self.optimizer = optim.Adam(varlist, lr=1e-2, weight_decay=0.0001)

        self.best_loss_total = np.inf
        self.epochs_completed = 0
        self.best_model_fname = None
        if self.best_model_fname is not None:
            self.vposer_model.load_state_dict(torch.load(self.best_model_fname, map_location=self.comp_device))

        self.ske = Skeleton(skeleton_path=skeleton_path)
        self.ske.to(self.comp_device)
        self.default_trans = torch.zeros(3).view(1,3).to(self.comp_device)

    def train(self):
        self.vposer_model.train()
        save_every_it = len(self.ds_train) / 4
        train_loss_dict = {}
        for it, dorig in enumerate(self.ds_train):
            dorig = {k: dorig[k].to(self.comp_device) for k in dorig.keys()}

            self.optimizer.zero_grad()
            drec = self.vposer_model(dorig['pose_aa'], output_type='aa')
            loss_total, cur_loss_dict = self.compute_loss(dorig, drec)
            loss_total.backward()
            self.optimizer.step()

            train_loss_dict = {k: train_loss_dict.get(k, 0.0) + v.item() for k, v in cur_loss_dict.items()}
            if it % (save_every_it + 1) == 0:
                cur_train_loss_dict = {k: v / (it + 1) for k, v in train_loss_dict.items()}
                print("Training Iteration: {}, Loss: {}".format(it, cur_train_loss_dict))

        train_loss_dict = {k: v / len(self.ds_train) for k, v in train_loss_dict.items()}
        return train_loss_dict

    def evaluate(self, split_name= 'vald'):
        self.vposer_model.eval()
        eval_loss_dict = {}
        data = self.ds_val if split_name == 'vald' else self.ds_test
        with torch.no_grad():
            for dorig in data:
                dorig = {k: dorig[k].to(self.comp_device) for k in dorig.keys()}
                drec = self.vposer_model(dorig['pose_aa'], output_type='aa')
                _, cur_loss_dict = self.compute_loss(dorig, drec)
                eval_loss_dict = {k: eval_loss_dict.get(k, 0.0) + v.item() for k, v in cur_loss_dict.items()}

        eval_loss_dict = {k: v / len(data) for k, v in eval_loss_dict.items()}
        return eval_loss_dict

    def compute_loss(self, dorig, drec):
        q_z = torch.distributions.normal.Normal(drec['mean'], drec['std'])

        prec = drec['pose_aa'].to(self.comp_device)
        porig = dorig['pose_aa'].to(self.comp_device)

        device = dorig['pose_aa'].device
        dtype = dorig['pose_aa'].dtype

        batchnum = prec.shape[0]

        trans = self.default_trans.repeat(batchnum,1)

        joint_rec = self.ske(prec, trans)
        joint_rig = self.ske(porig, trans)

        loss_joint_rec = (1. - 5e-3) * torch.mean(torch.abs(joint_rec - joint_rig))

        # KL loss
        p_z = torch.distributions.normal.Normal(
            loc=torch.tensor(np.zeros([self.batch_size, self.latentD]), requires_grad=False).to(device).type(dtype),
            scale=torch.tensor(np.ones([self.batch_size, self.latentD]), requires_grad=False).to(device).type(dtype))
        loss_kl = 5e-3 * torch.mean(torch.sum(torch.distributions.kl.kl_divergence(q_z, p_z), dim=[1]))

        ## Archive of losses
        # loss_rec = (1. - self.ps.kl_coef) * torch.mean(torch.sum(torch.pow(dorig - prec, 2), dim=[1, 2, 3]))
        # R = prec.view([batch_size, n_joints, 3, 3])
        # R_T = torch.transpose(R, 2, 3)
        # R_eye = torch.tensor(np.tile(np.eye(3,3).reshape(1,1,3,3), [batch_size, n_joints, 1, 1]), dtype=dtype, requires_grad = False).to(device)
        # loss_ortho = self.ps.ortho_coef * torch.mean(torch.sum(torch.pow(torch.matmul(R, R_T) - R_eye,2),dim=[1,2,3]))
        #
        # det_R = torch.transpose(torch.stack([determinant_3d(R[:,jIdx,...]) for jIdx in range(n_joints)]),0,1)
        #
        # one = torch.tensor(np.ones([batch_size, n_joints]), dtype = dtype, requires_grad = False).to(device)
        # loss_det1 = self.ps.det1_coef * torch.mean(torch.sum(torch.abs(det_R - one), dim=[1]))

        loss_dict = {'loss_kl': loss_kl,
                    'loss_joint_rec': loss_joint_rec
                     }

        if self.vposer_model.training and self.epochs_completed < 10:
            loss_dict['loss_pose_rec'] = (1. - 5e-3) * torch.mean(torch.sum(torch.pow(porig - prec, 2), dim=[1, 2, 3]))

        loss_total = torch.stack(list(loss_dict.values())).sum()
        loss_dict['loss_total'] = loss_total

        return loss_total, loss_dict

    def perform_training(self, num_epochs=None, message=None):
        starttime = datetime.now().replace(microsecond=0)
        if num_epochs is None: num_epochs = 500

        print(
            'Started Training at %s for %d epochs' % (datetime.strftime(starttime, '%Y-%m-%d_%H:%M:%S'), num_epochs))

        prev_lr = np.inf
        scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=int(num_epochs // 3), gamma=0.5)
        self.best_loss_total = np.inf
        loop = tqdm.tqdm(range(1, num_epochs + 1))
        for epoch_num in loop:
            print("Started Training Epoch {}".format(epoch_num))
            cur_lr = self.optimizer.param_groups[0]['lr']
            if cur_lr != prev_lr:
                print('--- Optimizer learning rate changed from %.2e to %.2e ---' % (prev_lr, cur_lr))
                prev_lr = cur_lr
            self.epochs_completed += 1
            train_loss_dict = self.train()
            eval_loss_dict = self.evaluate()
            scheduler.step()

            with torch.no_grad():
                print("Eval Training Epoch {}".format(epoch_num))
                if eval_loss_dict['loss_total'] < self.best_loss_total:
                    self.best_model_fname = os.path.join('snapshots', 'E%03d.pt' % (
                    self.epochs_completed))
                    self.best_loss_total = eval_loss_dict['loss_total']
                    torch.save(self.vposer_model.state_dict(), self.best_model_fname)
                    print("Loss {} is less, save model to {}".format(self.best_loss_total, self.best_model_fname))
                else:
                    print("Loss {} is larger, skip".format(eval_loss_dict['loss_total']))

        endtime = datetime.now().replace(microsecond=0)

        print('Finished Training at %s\n' % (datetime.strftime(endtime, '%Y-%m-%d_%H:%M:%S')))
        print('Training done in %s! Best val total loss achieved: %.2e\n' % (endtime - starttime, self.best_loss_total))
        print('Best model path: %s\n' % self.best_model_fname)

def run_vposer_trainer(datapath, bodymodel_path):
    vp_trainer = VPoserTrainer(datapath, bodymodel_path)

    vp_trainer.perform_training()

    test_loss_dict = vp_trainer.evaluate(split_name='test')

    print('Final loss on test set is %s' % (' | '.join(['%s = %.2e' % (k, v) for k, v in test_loss_dict.items()])))

if __name__ == '__main__':
    run_vposer_trainer("data/train/pose", "data/skeleton.pt")
