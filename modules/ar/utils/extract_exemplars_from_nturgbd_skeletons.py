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
             "S001C003P008R001A055",
             "S018C003P008R001A061",
             "S018C003P008R001A067",
             "S018C003P008R001A073",
             "S018C003P008R001A079",
             "S018C003P008R001A085",
             "S018C003P008R001A091",
             "S018C003P008R001A097",
             "S018C003P008R001A103",
             "S018C003P008R001A109",
             "S018C003P008R001A115"
             ]

test_classes = ["A1", "A7", "A13", "A19", "A25", "A31", "A37", "A43", "A49",
                "A55",
                "A61", "A67", "A73", "A79", "A85",
                "A91", "A97", "A103",
                "A109", "A115"
                ]

exemplars = [elem + '.skeleton' for elem in exemplars]
in_dataset_path = ["D:\\datasets\\useless\\nturgb+d_skeletons-1-17", "D:\\datasets\\useless\\nturgb+d_skeletons-18-32"]
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

    # Get conversion class id -> class label
    with open(classes_path, "r", encoding='utf-8') as f:
        classes = f.readlines()
    class_dict = {}
    for c in classes:
        index, name, _ = c.split(".")
        name = name.strip().replace(" ", "_").replace("/", "-").replace("’", "")
        if index in test_classes:
            class_dict[index] = name

    # Create output directories ( and remove old one if presents)
    for value in list(class_dict.values()):
        if os.path.exists(os.path.join(out_dataset_path, value)):
            shutil.rmtree(os.path.join(out_dataset_path, value))
        os.mkdir(os.path.join(out_dataset_path, value))

    for file in exemplars:
        # Retrieve class name (between A and _ es 'S001C001P001R001A001_rgb.avi'
        class_id = int(file.split("A")[-1].split(".")[0])  # take the integer of the class
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
        pose_sequences = {}
        offset = 1
        for i in range(n_frame):

            body_count = int(lines[offset].strip())
            offset += 1  # go to info line

            for k in range(body_count):

                body_id = lines[offset].split()[0]
                offset += 1  # go to joint number line
                joint_count = int(lines[offset].strip())
                offset += 1  # Advance to coordinates

                pose = []
                for _ in range(joint_count):
                    pose.append(lines[offset].strip().split(' ')[:3])
                    offset += 1  # Next line
                if body_id in pose_sequences.keys():
                    pose_sequences[body_id].append(pose)
                else:
                    pose_sequences[body_id] = list([pose])
                # if body_count > 1:
                #     offset -= 1  # Skip

        if len(pose_sequences) == 0:
            print("No skeleton found!")
            continue
        pose_sequence = pose_sequences[max(pose_sequences, key=lambda item: len(pose_sequences[item]))]
        if len(pose_sequence) < n:
            print("Skipping (not enough poses)")
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
