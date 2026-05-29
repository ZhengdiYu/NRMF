''










import os, sys, argparse; sys.path.insert(0, ''); sys.path.insert(0, 'motionlib')
import numpy as np, torch
from gen.dataset import process_npz, FPS
from gen.contact_net import ContactNet
from gen.train_contact import derive_om, add_noise, f1, FEET
from utils.transforms import (axis_angle_to_quaternion as aa2q, quaternion_hamilton_product as qmul,
                              quaternion_conjugate as qconj, quaternion_log_map as qlog)
from kinematics import fk_joints, LEG
from noise import rohm_noise_aa
from humor.body_model.body_model import BodyModel
dev = 'cuda'
ROOT = 'data'
BMP = 'body_models/smplh/neutral/model.npz'
CKP = 'checkpoints/contact.pt'

SEQS = [('run_walk',  ROOT + '/Transitions_mocap/mazen_c3d/run_walk_poses_253_frames_30_fps.npz'),
        ('turn180',   ROOT + '/Transitions_mocap/mazen_c3d/JOOF_turntwist180_poses_151_frames_30_fps.npz'),
        ('jumpjacks', ROOT + '/SSM_synced/20160930_50032/jumping_jacks_sync_poses_94_frames_30_fps.npz'),
        ('fwdwalk',   ROOT + '/SFU/0007/0007_Walking001_poses_786_frames_30_fps.npz')]


def load_net():
    ck = torch.load(CKP, map_location=dev); ar = ck['args']
    net = ContactNet(d_model=ar['d_model'], n_layers=ar['layers'], n_heads=ar['heads'], max_len=ar['window'] + 8).to(dev)
    net.load_state_dict(ck['model']); net.eval()
    return net, ck['om_std'].to(dev), ar['window']


def height_contact(q_canon, trans, betas, bm, thresh=0.08):
    ''
    J = fk_joints(q_canon, betas, bm, trans); fz = J[:, FEET, 2]
    return (fz - fz.min() < thresh).float()


def predicted_contact(net, q_canon, om_std):
    om = derive_om(q_canon.unsqueeze(0), om_std)
    return net.predict(q_canon.unsqueeze(0), om)[0]
