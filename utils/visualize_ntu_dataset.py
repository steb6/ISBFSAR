import json
import os
import pickle
import time

import numpy as np
from utils.output import PosePrinter

in_dataset_path = "D:\\nturgbd_metrabs"
skeleton = "smpl+head_30"

# Load conversions
with open('assets/skeleton_types.pkl', "rb") as input_file:
    skeleton_types = pickle.load(input_file)

# Load visualizer
vis = PosePrinter(640, 480, just_pose=True)

for root, dirs, files in os.walk(in_dataset_path):
    print(root)
    for file in files:
        path = os.path.join(root, file)
        with open(path, 'rb') as pose_file:
            res = json.load(pose_file)
        while True:  # TODO REMOVE
            for elem in res:
                elem = np.array(elem)
                pred3d = elem[skeleton_types[skeleton]['indices']]
                pred3d -= pred3d[0]
                edges = skeleton_types[skeleton]['edges']
                vis.print_pose(pred3d, edges)
                time.sleep(0.1)

