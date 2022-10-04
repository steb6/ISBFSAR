import pickle
import os
import cv2
import numpy as np
from modules.hpe.hpe import HumanPoseEstimator
from utils.matplotlib_visualizer import MPLPosePrinter
from utils.params import MetrabsTRTConfig, RealSenseIntrinsics
import shutil
from tqdm import tqdm

in_dataset_path = "D:\\datasets\\nturgbd\\videos"
out_dataset_path = "D:\\datasets\\NTURGBD_to_YOLO_METRO_exemplars"
classes_path = "assets/nturgbd_classes.txt"
n = 16

exemplars = ["S001C003P008R001A001",
             "S001C003P008R001A007",
             "S001C003P008R001A013",
             "S001C003P008R001A019",
             "S001C003P008R001A025",
             "S001C003P008R001A031",
             "S001C003P008R001A037",
             "S001C003P008R001A043",
             "S001C003P008R001A049",
             # "S001C003P008R001A055",  # Two people
             "S018C003P008R001A061",
             "S018C003P008R001A067",
             "S018C003P008R001A073",
             "S018C003P008R001A079",
             "S018C003P008R001A085",
             "S018C003P008R001A091",
             "S018C003P008R001A097",
             "S018C003P008R001A103",
             # "S018C003P008R001A109",  # Two people
             # "S018C003P008R001A115"  # Two people
             ]

VIS_DEBUG = False

if __name__ == "__main__":

    skeleton = 'smpl+head_30'
    with open('assets/skeleton_types.pkl', "rb") as input_file:
        skeleton_types = pickle.load(input_file)
    edges = skeleton_types[skeleton]['edges']

    model = HumanPoseEstimator(MetrabsTRTConfig(), RealSenseIntrinsics(), just_box=False)
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

    # Create output directories
    if os.path.exists(out_dataset_path):
        shutil.rmtree(out_dataset_path)
    os.mkdir(out_dataset_path)

    # Iterate all videos
    for example in tqdm(exemplars):

        # Retrieve class name (between A and _ es 'S001C001P001R001A001_rgb.avi'
        class_id = int(example.split("A")[1].split("_")[0])  # take the integer of the class
        class_id = "A" + str(class_id)
        class_name = class_dict[class_id]

        # Create output directory
        output_path = os.path.join(out_dataset_path, class_name)
        os.mkdir(output_path)
        output_path = os.path.join(output_path, "0")
        os.mkdir(output_path)

        # Read video
        dir_tag = example.split("C")[0].lower()
        if int(dir_tag[1:]) < 18:
            dir_tag = dir_tag[1:]
        dir_name = "nturgb+d_rgb_" + dir_tag
        full = os.path.join(in_dataset_path, dir_name, example+"_rgb.avi")
        video = cv2.VideoCapture(full)
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
        images = []
        res = np.zeros([30, 3])  # So if the first pose is none, there is no error

        for i, frame in enumerate(frames):
            frame = frame[:, 240:-240, :]
            frame = cv2.resize(frame, (640, 480))

            if VIS_DEBUG:
                cv2.imshow("Original", frame)  # TODO VIS DEBUG

            res = model.estimate(frame)
            if res is not None:
                pose, edges, bbox = res['pose'], res["edges"], res["bbox"]
            else:
                pose, edges, bbox = pose_, edges_, box_
            pose_, edges_, box_ = pose, edges, bbox

            pose = pose - pose[0]
            poses.append(pose)

            x1, x2, y1, y2 = bbox
            xm = int((x1 + x2) / 2)
            ym = int((y1 + y2) / 2)
            l = max(xm - x1, ym - y1)
            frame = frame[(ym - l if ym - l > 0 else 0):(ym + l), (xm - l if xm - l > 0 else 0):(xm + l)]
            images.append(frame)

            if VIS_DEBUG:
                cv2.imshow("Cropped", cv2.resize(frame, (224, 224)))  # TODO VIS DEBUG
                vis.clear()  # TODO VIS DEBUG
                vis.print_pose(pose, edges)  # TODO VIS DEBUG
                vis.sleep(0.01)  # TODO VIS DEBUG

        # Save result
        for i in range(len(poses)):
            sample_path = os.path.join(output_path, f"{i}")
            with open(sample_path+'.pkl', "wb") as f:
                pickle.dump(poses[i], f)
            cv2.imwrite(sample_path+'.png', images[i])
        assert (i+1) == n
