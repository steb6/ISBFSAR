import pickle
import os
import cv2
import shutil
import numpy as np
from tqdm import tqdm
from modules.hpe.hpe import HumanPoseEstimator
from utils.matplotlib_visualizer import MPLPosePrinter
from utils.params import MetrabsTRTConfig, RealSenseIntrinsics
import pycuda.autoinit

in_dataset_path = "D:\\datasets\\nturgbd\\videos"
out_dataset_path = "D:\\datasets\\NTURGBD_to_YOLO_METRO_122"
classes_path = "assets/nturgbd_classes.txt"
n = 16

VIS_DEBUG = False

if __name__ == "__main__":

    # raise Exception("REMOVE THIS LINE TO ERASE CURRENT DATASET")  # TODO ADD

    skeleton = 'smpl+head_30'
    with open('assets/skeleton_types.pkl', "rb") as input_file:
        skeleton_types = pickle.load(input_file)
    edges = skeleton_types[skeleton]['edges']

    model = HumanPoseEstimator(MetrabsTRTConfig(), RealSenseIntrinsics(), just_box=False)
    if VIS_DEBUG:
        vis = MPLPosePrinter()  # TODO VIS DEBUG

    # Count total number of files
    total = int(sum([len(files) for r, d, files in os.walk(in_dataset_path)]) * (1 - 26/120))

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
    for i, value in enumerate(list(class_dict.values())):
        if 0 <= i <= 49-1 or 61-1 <= i <= 105-1:
            if os.path.exists(os.path.join(out_dataset_path, value)):
                shutil.rmtree(os.path.join(out_dataset_path, value))
            os.mkdir(os.path.join(out_dataset_path, value))

    # In this way, we can continue the extraction without losing data
    class_offset = {}
    for c in class_dict.values():
        class_offset[c] = 0

    # Iterate all videos
    with tqdm(total=total) as progress_bar:
        for root, dirs, files in os.walk(in_dataset_path):

            for file in files:
                try:
                    # Retrieve class name (between A and _ es 'S001C001P001R001A001_rgb.avi'
                    class_id = int(file.split("A")[1].split("_")[0])  # take the integer of the class
                    class_id = "A" + str(class_id)
                    class_name = class_dict[class_id]

                    # Skip if two person are involved
                    if 50-1 <= list(class_dict.keys()).index(class_id) <= 60-1 or \
                            106 - 1 <= list(class_dict.keys()).index(class_id) <= 120 - 1:
                        continue

                    # Compute offset
                    output_path = os.path.join(out_dataset_path, class_name)
                    offset = len(os.listdir(output_path))
                    class_offset[class_name] += 1
                    if class_offset[class_name] < offset:  # Already extracted
                        progress_bar.update()
                        continue
                    output_path = os.path.join(output_path, str(offset))

                    # Read video
                    full = os.path.join(root, file)
                    video = cv2.VideoCapture(full)
                    frames = []
                    ret, frame = video.read()
                    while ret:
                        frames.append(frame)
                        ret, frame = video.read()
                    if len(frames) < n:
                        raise Exception("Video", full, "didn't have enough frames")

                    # Select just n frames
                    n_frames = len(frames) - (len(frames) % n)
                    if n_frames == 0:
                        continue
                    indices = list(range(0, n_frames, int(n_frames / n)))
                    frames = [frames[i] for i in indices]

                    # Iterate over all frames
                    poses = []
                    images = []
                    res = np.zeros([122, 3])  # So if the first pose is none, there is no error

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
                    os.mkdir(output_path)
                    for i in range(len(poses)):
                        sample_path = os.path.join(output_path, f"{i}")
                        with open(sample_path+'.pkl', "wb") as f:
                            pickle.dump(poses[i], f)
                        cv2.imwrite(sample_path+'.png', images[i])
                    assert (i+1) == n

                    progress_bar.update()
                except Exception as e:
                    print(e.__str__())
