import torch
from torch.autograd import Variable
import numpy as np


def normalize_last_columns(m):
    """
    m: Tensor (points x coords)
    """
    for i in range(m.shape[1]):
        # Normalize along i dimension
        y_std = (m[..., i] - m[..., i].min(axis=0)[0].unsqueeze(-1).repeat(1, m.size(-2))) / \
                (m[..., i].max(axis=0)[0].unsqueeze(-1).repeat(1, m.size(-2)) -
                 m[..., i].min(axis=0)[0].unsqueeze(-1).repeat(1, m.size(-2)))
        m[..., 0] = y_std * (1 - 0) + 0

    return m


def oneHotVectorize(targets, size=1):
    oneHotTarget = torch.zeros(targets.size()[0], size)

    for o in range(targets.size()[0]):
        oneHotTarget[o][targets[o].item()] = 1

    oneHotTarget = oneHotTarget.to(targets.device)
    oneHotTarget = Variable(oneHotTarget, requires_grad=False)
    return oneHotTarget


def th_delete(tensor, indices):
    mask = torch.ones(tensor.numel(), dtype=torch.bool)
    mask[indices] = False
    return tensor[mask]


def two_poses_movement(pose1, pose2):
    return np.sum(np.sqrt(np.square(pose1[:, 0] - pose2[:, 0]) +
                          np.square(pose1[:, 1] - pose2[:, 1]) +
                          np.square(pose1[:, 2] - pose2[:, 2])))


def two_poses_movement_torch(pose1, pose2):
    return torch.sum(torch.sqrt(torch.square(pose1[:, 0] - pose2[:, 0]) +
                                torch.square(pose1[:, 1] - pose2[:, 1]) +
                                torch.square(pose1[:, 2] - pose2[:, 2])))
