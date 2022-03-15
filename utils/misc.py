import torch
import numpy as np


def two_poses_movement(pose1, pose2):
    return np.sum(np.sqrt(np.square(pose1[:, 0] - pose2[:, 0]) +
                          np.square(pose1[:, 1] - pose2[:, 1]) +
                          np.square(pose1[:, 2] - pose2[:, 2])))


def two_poses_movement_torch(pose1, pose2):
    return torch.sum(torch.sqrt(torch.square(pose1[:, 0] - pose2[:, 0]) +
                                torch.square(pose1[:, 1] - pose2[:, 1]) +
                                torch.square(pose1[:, 2] - pose2[:, 2])))
