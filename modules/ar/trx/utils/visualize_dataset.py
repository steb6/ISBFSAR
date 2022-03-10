import json
import os
import pickle
import time

import numpy as np
import torch
from tqdm import tqdm
from dataloader import MetrabsData
from utils.output import PosePrinter

if __name__ == "__main__":
    data_path = "D:\\nturgbd_metrabs" if 'Users' in os.getcwd() else "nturgbd_metrabs"
    data = MetrabsData(data_path, k=5, mode='train', n_task=10000, debug_classes=True)
    train_loader = torch.utils.data.DataLoader(data, batch_size=1, num_workers=2)

    vis = PosePrinter(640, 480, just_pose=True)

    skeleton = 'smpl+head_30'
    with open('assets/skeleton_types.pkl', "rb") as input_file:
        skeleton_types = pickle.load(input_file)
    edges = skeleton_types[skeleton]['edges']

    for elem in tqdm(train_loader):
        torch.set_grad_enabled(True)

        support_set, target_set, support_labels, target_label = elem

        print("TARGET")
        print(data.classes[target_label])
        for pose in target_set[0].detach().cpu().numpy():
            vis.print_pose(pose, edges)
            time.sleep(0.1)
        time.sleep(1)
        for i, pose in enumerate(support_set[0].detach().cpu().numpy()):
            print("SUPPORT SET")
            print(data.classes[support_labels[0, i].item()])
            for action in pose:
                vis.print_pose(action, edges)
                time.sleep(0.1)
            time.sleep(1)
