import json
import os
import cv2
import shutil
import numpy as np
from tqdm import tqdm
from models.metrabs_trt.hpe import HumanPoseEstimator

in_dataset_path = "D:\\nturgbd"
out_dataset_path = "D:\\nturgbd_metrabs"
classes_path = "assets\\nturgbd_classes.txt"
n = 16

if __name__ == "__main__":

    raise Exception("REMOVE THIS LINE TO ERASE CURRENT DATASET")

    # Load model
    args = {
        "yolo_engine_path": 'models/metrabs_trt/models/trts/yolo16.trt',
        "bbone_engine_path": 'models/metrabs_trt/models/trts/bbone16.trt',
        "heads_engine_path": 'models/metrabs_trt/models/signatured/metrab_head',
        "expand_joints_path": 'models/metrabs_trt/assets/32_to_122.npy',
        "skeleton_types_path": 'models/metrabs_trt/assets/skeleton_types.pkl'
    }
    cam_args = {'fx': 384.025146484375, 'fy': 384.025146484375, 'ppx': 319.09661865234375,
                'ppy': 237.75723266601562,
                'width': 640, 'height': 480}
    model = HumanPoseEstimator(*args.values(), cam_args, yolo_port=7000, bbone_port=7005, heads_port=7010)

    # Count total number of files
    total = sum([len(files) for r, d, files in os.walk(in_dataset_path)])

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
    for value in class_dict.values():
        os.mkdir(os.path.join(out_dataset_path, value))

    # TODO if we have only the first 60 classes, erase the other 60
    for i, value in enumerate(class_dict.values()):
        if i >= 60:
            to_erase = os.path.join(out_dataset_path, value)
            shutil.rmtree(to_erase)
    # exit()

    # Iterate all videos
    with tqdm(total=total) as progress_bar:
        for root, dirs, files in os.walk(in_dataset_path):
            for file in files:
                full = os.path.join(root, file)
                video = cv2.VideoCapture(full)
                frames = []
                ret, frame = video.read()
                while ret:
                    frames.append(frame)
                    ret, frame = video.read()

                # Select just n frames
                n_frames = len(frames) - (len(frames) % n)
                indices = list(range(0, n_frames, int(n_frames / n)))
                frames = [frames[i] for i in indices]

                # Retrieve class name (between A and _ es 'S001C001P001R001A001_rgb.avi'
                class_id = int(file.split("A")[1].split("_")[0])  # take the integer of the class
                class_id = "A" + str(class_id)
                class_name = class_dict[class_id]

                # Save results (SKELETON)
                output_path = os.path.join(out_dataset_path, class_name)
                offset = sum([len(files) for r, d, files in os.walk(output_path)])
                output_path = os.path.join(output_path, str(offset) + '.json')

                poses = []

                for i, frame in enumerate(frames):
                    res = model.estimate(frame, just_root=True)[0]
                    poses.append(res)

                poses = np.array(poses).tolist()

                with open(output_path, "w") as f:
                    json.dump(poses, f)

                assert (i+1) == n

                # Save results (IMAGES)
                # output_path = os.path.join(out_dataset_path, class_name)
                # offset = len(next(os.walk(output_path))[1])
                # output_path = os.path.join(output_path, str(offset))
                # os.mkdir(output_path)
                # for i, frame in enumerate(frames):
                #     final = os.path.join(output_path, str(i)) + '.jpg'
                #     cv2.imwrite(final, frame)
                #     # print(final)
                # assert (i+1) == n

                progress_bar.update()
