#!/usr/bin/python3

import math

# image
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480

# open pose
JOINTS_POSE = [0, 15, 16, 17, 18]
JOINTS_FACE = [36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 68, 69]
#JOINTS_FACE = [36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]  # without pupils
NUM_JOINTS = len(JOINTS_FACE) + len(JOINTS_POSE)

conf_threshold = 0.0


def joint_set(p, c):
    return (p[0] != 0.0 or p[1] != 0.0) and c >= conf_threshold


def compute_centroid(points):
    mean_x = np.mean([p[0] for p in points])
    mean_y = np.mean([p[1] for p in points])

    return [mean_x, mean_y]


def get_eye_bbox_openpose(pose, conf_pose):
    n_joints_set = [pose[joint] for joint in JOINTS_POSE if joint_set(pose[joint], conf_pose[joint])]
    if n_joints_set:
        min_x = min([joint[0] for joint in n_joints_set])
        max_x = max([joint[0] for joint in n_joints_set])
        min_x -= (max_x - min_x) * 0.4
        max_x += (max_x - min_x) * 0.4

        min_y = min([joint[1] for joint in n_joints_set])
        max_y = max([joint[1] for joint in n_joints_set])
        min_y -= (max_y - min_y) * 0.4
        max_y += (max_y - min_y) * 0.4

        min_x = math.floor(max(0, min(min_x, IMAGE_WIDTH)))
        max_x = math.floor(max(0, min(max_x, IMAGE_WIDTH)))
        min_y = math.floor(max(0, min(min_y, IMAGE_HEIGHT)))
        max_y = math.floor(max(0, min(max_y, IMAGE_HEIGHT)))

        return min_x, min_y, max_x, max_y
    else:
        print("Joint set empty!")
        return None, None, None, None


def get_face_bbox_openpose(pose, conf_pose):
    n_joints_set = [pose[joint] for joint in JOINTS_POSE if joint_set(pose[joint], conf_pose[joint])]
    if n_joints_set:
        centroid = compute_centroid(n_joints_set)

        min_x = min([joint[0] for joint in n_joints_set])
        max_x = max([joint[0] for joint in n_joints_set])
        min_x -= (max_x - min_x) * 0.4
        max_x += (max_x - min_x) * 0.4

        width = max_x - min_x

        min_y = centroid[1] - (width/3)*2
        max_y = centroid[1] + (width/3)*2

        min_x = math.floor(max(0, min(min_x, IMAGE_WIDTH)))
        max_x = math.floor(max(0, min(max_x, IMAGE_WIDTH)))
        min_y = math.floor(max(0, min(min_y, IMAGE_HEIGHT)))
        max_y = math.floor(max(0, min(max_y, IMAGE_HEIGHT)))

        return min_x, min_y, max_x, max_y
    else:
        print("Joint set empty!")
        return None, None, None, None

