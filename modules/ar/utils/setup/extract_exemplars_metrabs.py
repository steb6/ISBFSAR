import pickle
import os
import cv2
import shutil
import numpy as np
from tqdm import tqdm
from modules.hpe.hpe import HumanPoseEstimator
from utils.matplotlib_visualizer import MPLPosePrinter
from utils.params import MetrabsTRTConfig, RealSenseIntrinsics

exemplars = ["S001C003P008R001A001",  # LAST WAS 1
             "S001C003P008R001A007",
             "S001C003P008R001A013",
             "S001C003P008R001A019",
             "S001C003P008R001A025",
             "S001C003P008R001A031",
             "S001C003P008R001A037",
             "S001C003P008R001A043",
             "S001C003P008R001A049",
             # "S001C003P008R001A055",
             "S018C003P008R001A061",
             "S018C003P008R001A067",
             "S018C003P008R001A073",
             "S018C003P008R001A079",
             "S018C003P008R001A085",
             "S018C003P008R001A091",
             "S018C003P008R001A097",
             "S018C003P008R001A103",
             # "S018C003P008R001A109",
             # "S018C003P008R001A115"
             ]

test_classes = ["A1", "A7", "A13", "A19", "A25", "A31", "A37", "A43", "A49",
                # "A55",
                "A61", "A67", "A73", "A79", "A85",
                "A91", "A97", "A103",
                # "A109", "A115"
                ]

VIS_DEBUG = True

exemplars = [elem + '.skeleton' for elem in exemplars]
in_dataset_path = "D:\\datasets\\useless\\nturgbd"
out_dataset_path = "D:\\datasets\\metrabs_trx_skeletons_exemplars"
classes_path = "assets/nturgbd_classes.txt"
n = 16


if __name__ == "__main__":

    # raise Exception("REMOVE THIS LINE TO ERASE CURRENT DATASET")  # TODO ADD
    if VIS_DEBUG:
        vis = MPLPosePrinter()  # TODO VIS DEBUG

    # Get conversion class id -> class label
    with open(classes_path, "r", encoding='utf-8') as f:
        classes = f.readlines()
    class_dict = {}
    for c in classes:
        index, name, _ = c.split(".")
        name = name.strip().replace(" ", "_").replace("/", "-").replace("’", "")
        class_dict[index] = name

    # Create output directories (ONLY THE MISSING ONES)  # TODO CAREFUL, ERASE WHAT DONE BEFORE
    # for value in list(class_dict.values())[60:]:
    #     shutil.rmtree(os.path.join(out_dataset_path, value))
    #     os.mkdir(os.path.join(out_dataset_path, value))

    # Iterate all videos
    paths = []
    for root, dirs, files in os.walk(in_dataset_path):
        for name in files:
            paths.append(os.path.join(root, name))
        for name in dirs:
            paths.append(os.path.join(root, name))

    skeleton = 'smpl+head_30'
    with open('assets/skeleton_types.pkl', "rb") as input_file:
        skeleton_types = pickle.load(input_file)
    edges = skeleton_types[skeleton]['edges']

    model = HumanPoseEstimator(MetrabsTRTConfig(), RealSenseIntrinsics())

    for example in exemplars:
        video_name = example.split('.')[0]
        video_path = [p for p in paths if video_name in p][0]
        # Retrieve class name (between A and _ es 'S001C001P001R001A001_rgb.avi'
        class_id = int(video_name.split("A")[1].split("_")[0])  # take the integer of the class
        class_id = "A" + str(class_id)
        class_name = class_dict[class_id]

        # Check if output path already exists
        output_path = os.path.join(out_dataset_path, class_name)
        offset = sum([len(files) for r, d, files in os.walk(output_path)])
        output_path = os.path.join(output_path, str(offset) + '.pkl')

        # Read video
        video = cv2.VideoCapture(video_path)
        frames = []
        ret, frame = video.read()
        while ret:
            frames.append(frame)
            ret, frame = video.read()
        if len(frames) < n:
            continue

        # Select just n frames
        n_frames = len(frames) - (len(frames) % n)
        if n_frames == 0:
            continue
        indices = list(range(0, n_frames, int(n_frames / n)))
        frames = [frames[i] for i in indices]

        # Iterate over all frames
        poses = []
        res = np.zeros([30, 3])  # So if the first pose is none, there is no error

        for i, frame in enumerate(frames):
            frame = frame[:, 240:-240, :]
            frame = cv2.resize(frame, (640, 480))
            new_res = model.estimate(frame)[0]
            res = res if new_res is None else new_res
            res = res - res[0]
            poses.append(res)

            if VIS_DEBUG:
                cv2.imshow("", frame)  # TODO VIS DEBUG
                vis.clear()  # TODO VIS DEBUG
                vis.print_pose(res, edges)  # TODO VIS DEBUG
                vis.sleep(0.01)  # TODO VIS DEBUG

        # Save result
        poses = np.array(poses)
        os.mkdir(os.path.join(out_dataset_path, class_name))
        with open(output_path, "wb") as f:
            pickle.dump(poses, f)
        assert (i+1) == n
