import torch
from tqdm import tqdm
from dataloader import MetrabsData
from utils.matplotlib_visualizer import MPLPosePrinter as PosePrinter
from utils.params import TRXConfig
import numpy as np

if __name__ == "__main__":
    data_path = TRXConfig().data_path
    data = MetrabsData(data_path, k=5, n_task=10000, return_true_class_index=True)
    data_loader = torch.utils.data.DataLoader(data, batch_size=1, num_workers=0)

    vis = PosePrinter()

    # Get edges of the skeleton
    edges = np.array([[1, 2],
                      [2, 1],
                      [3, 21],
                      [4, 3],
                      [5, 21],
                      [6, 5],
                      [7, 6],
                      [8, 7],
                      [9, 21],
                      [10, 9],
                      [11, 10],
                      [12, 11],
                      [13, 1],
                      [14, 13],
                      [15, 14],
                      [16, 15],
                      [17, 1],
                      [18, 17],
                      [19, 18],
                      [20, 19],
                      [21, 2],
                      [22, 8],
                      [23, 8],
                      [24, 12],
                      [25, 12]])

    for elem in tqdm(data_loader):
        support_set, target_set, support_labels, target_label = elem
        vis.set_title("TARGET: {}".format(data.classes[target_label]))
        for pose in target_set[0].detach().cpu().numpy():
            vis.clear()
            vis.print_pose(pose, edges)
            vis.sleep(0.01)
        # time.sleep(1)
        for i, pose in enumerate(support_set[0].detach().cpu().numpy()):
            vis.set_title("SUPPORT SET: {}".format(data.classes[support_labels[0, i].item()]))
            for action in pose:
                vis.clear()
                vis.print_pose(action, edges)
                vis.sleep(0.01)
            # vis.sleep(1)
