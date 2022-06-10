import pickle
import os
import shutil
import numpy as np
from tqdm import tqdm
from utils.matplotlib_visualizer import MPLPosePrinter

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

exemplars = [elem + '.skeleton' for elem in exemplars]
in_dataset_path = ["D:\\datasets\\nturgb+d_skeletons-1-17", "D:\\datasets\\nturgb+d_skeletons-18-32"]
out_dataset_path = "D:\\datasets\\nturgbd_trx_skeletons_exemplars"
classes_path = "assets/nturgbd_classes.txt"
n = 16

VIS_DEBUG = True
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
                  [25, 12]]) - 1

if __name__ == "__main__":

    if VIS_DEBUG:
        vis = MPLPosePrinter()

    with open("assets/nturgbd_without_skeleton.txt", "r") as infile:
        missing_skeleton = infile.readlines()[3:]

    # Count total number of files (we remove 10 classes over 60 because those involves two person)
    total = 0
    for path in in_dataset_path:
        total += int(sum([len(files) for r, d, files in os.walk(path)]) * (1 - 16 / 60))

    # Get conversion class id -> class label
    with open(classes_path, "r", encoding='utf-8') as f:
        classes = f.readlines()
    class_dict = {}
    for c in classes:
        index, name, _ = c.split(".")
        name = name.strip().replace(" ", "_").replace("/", "-").replace("’", "")
        if index in test_classes:
            class_dict[index] = name
    # two_person = ['A50', 'A51', 'A52', 'A53', 'A54', 'A55', 'A56', 'A57', 'A58', 'A59', 'A60', 'A106', 'A107', 'A108',
    #               'A109', 'A110', 'A111', 'A112', 'A113', 'A114', 'A115', 'A116', 'A117', 'A118', 'A119', 'A120']
    # for elem in two_person:
    #     class_dict.pop(elem)

    # Create output directories ( and remove old one if presents)
    for value in list(class_dict.values()):
        if os.path.exists(os.path.join(out_dataset_path, value)):
            shutil.rmtree(os.path.join(out_dataset_path, value))
        os.mkdir(os.path.join(out_dataset_path, value))

    for file in exemplars:
        # Retrieve class name (between A and _ es 'S001C001P001R001A001_rgb.avi'
        class_id = int(file.split("A")[-1].split(".")[0])  # take the integer of the class
        if 50 <= class_id <= 60 or class_id >= 106:
            # print("Skipping two person (by id)")
            continue
        class_id = "A" + str(class_id)
        class_name = class_dict[class_id]

        # Check if output path already exists
        output_path = os.path.join(out_dataset_path, class_name)
        offset = sum([len(files) for r, d, files in os.walk(output_path)])
        output_path = os.path.join(output_path, str(offset) + '.pkl')

        # Read file
        lines = None
        for path in in_dataset_path:
            if os.path.exists(os.path.join(path, file)):
                with open(os.path.join(path, file), "r") as infile:
                    lines = infile.readlines()
        assert lines is not None

        n_frame = int(lines[0].strip())
        pose_sequence = []
        offset = 1
        for i in range(n_frame):
            body_count = int(lines[offset].strip())

            # If there are two person, ignore this frame
            if body_count != 1:
                # print("Skipping frame (found 2 skeletons)")
                continue

            joint_count = int(lines[offset + 2].strip())
            pose = []
            for j in range(joint_count):
                pose.append(lines[offset + 3 + j].strip().split(' ')[:3])
            pose_sequence.append(pose)
            offset += joint_count + 3

        if len(pose_sequence) < n:
            # print("Skipping (not enough poses)")
            continue

        # Select just n frames
        n_pose = len(pose_sequence) - (len(pose_sequence) % n)
        if n_pose == 0:
            continue
        indices = list(range(0, n_pose, int(n_pose / n)))
        pose_sequence = [pose_sequence[i] for i in indices]

        # Iterate over all frames
        pose_sequence = np.array(pose_sequence).astype(float)
        pose_sequence -= pose_sequence[:, 0, :][:, None, :]  # Center skeleton

        # Save result
        with open(output_path, "wb") as f:
            pickle.dump(pose_sequence, f)
        assert len(pose_sequence) == n

        # Visualize
        for frame in pose_sequence:
            if VIS_DEBUG:
                vis.set_title(class_name)
                vis.clear()
                vis.print_pose(frame, edges)
                vis.sleep(0.01)
