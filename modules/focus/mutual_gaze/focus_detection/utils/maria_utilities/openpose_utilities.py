#!/usr/bin/python3

import numpy as np
import json

# image
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480

# open pose
JOINTS_POSE = [0, 15, 16, 17, 18]
JOINTS_FACE = [36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 68, 69]
#JOINTS_FACE = [36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]  # without pupils
NUM_JOINTS = len(JOINTS_FACE) + len(JOINTS_POSE)

conf_threshold = 0.0


def compute_centroid(points):
    mean_x = np.mean([p[0] for p in points])
    mean_y = np.mean([p[1] for p in points])

    if mean_x > IMAGE_WIDTH:
        mean_x = IMAGE_WIDTH
    if mean_x < 0:
        mean_x = 0
    if mean_y > IMAGE_HEIGHT:
        mean_y = IMAGE_HEIGHT
    if mean_y < 0:
        mean_y = 0

    return [mean_x, mean_y]


def joint_set(p, c):
    return (p[0] != 0.0 or p[1] != 0.0) and c >= conf_threshold


def dist_2d(p1, p2):
    p1 = np.array(p1)
    p2 = np.array(p2)

    squared_dist = np.sum((p1 - p2)**2, axis=0)
    return np.sqrt(squared_dist)


def compute_head_face_centroid(pose, conf_pose, face, conf_face):
    n_joints_set = [pose[joint] for joint in JOINTS_POSE if joint_set(pose[joint], conf_pose[joint])]
    n_joints_set.extend([face[joint] for joint in JOINTS_FACE if joint_set(face[joint], conf_face[joint])])

    if len(n_joints_set) < 2:
        return None

    centroid = compute_centroid(n_joints_set)

    return centroid


def compute_head_face_features(pose, conf_pose, face, conf_face):

    n_joints_set = [pose[joint] for joint in JOINTS_POSE if joint_set(pose[joint], conf_pose[joint])]
    n_joints_set.extend([face[joint] for joint in JOINTS_FACE if joint_set(face[joint], conf_face[joint])])

    if len(n_joints_set) < 2:
        return None, None

    centroid = compute_centroid(n_joints_set)
    max_dist = max([dist_2d(j, centroid) for j in n_joints_set])

    new_repr_pose = [(np.array(pose[joint]) - np.array(centroid)) for joint in JOINTS_POSE]
    new_repr_face = ([(np.array(face[joint]) - np.array(centroid)) for joint in JOINTS_FACE])

    result = []

    for i in range(0, len(JOINTS_POSE)):

        if joint_set(pose[JOINTS_POSE[i]], conf_pose[JOINTS_POSE[i]]):
            result.append([new_repr_pose[i][0] / max_dist, new_repr_pose[i][1] / max_dist])
        else:
            result.append([0, 0])

    for i in range(0, len(JOINTS_FACE)):
        if joint_set(face[JOINTS_FACE[i]], conf_face[JOINTS_FACE[i]]):
            result.append([new_repr_face[i][0] / max_dist, new_repr_face[i][1] / max_dist])
        else:
            result.append([0, 0])

    flat_list = [item for sublist in result for item in sublist]

    for j in JOINTS_POSE:
        if conf_pose[j] >= conf_threshold:
            flat_list.append(conf_pose[j])
        else:
            flat_list.append(0)

    for j in JOINTS_FACE:
        if conf_face[j] >= conf_threshold:
            flat_list.append(conf_face[j])
        else:
            flat_list.append(0)

    return flat_list, centroid


def load_many_poses(data):
    poses = []
    confidences = []

    for person in data:
        poses.append(np.array(person)[:, 0:2])
        confidences.append(np.array(person)[:, 2])

    return poses, confidences


def load_many_faces(data):
    faces = []
    confidences = []

    for person in data:
        faces.append(np.array(person)[:, 0:2])
        confidences.append(np.array(person)[:, 2])

    return faces, confidences

########################################################

# read openpose data from the bottle sent by yarpOpenPose
def read_openpose_data(received_data):
    body = []
    face = []
    if received_data:
        received_data = received_data.get(0).asList()
        for i in range(0, received_data.size()):
            keypoints = received_data.get(i).asList()

            if keypoints:
                body_person = []
                face_person = []
                for y in range(0, keypoints.size()):
                    part = keypoints.get(y).asList()
                    if part:
                        if part.get(0).asString() == "Face":
                            for z in range(1, part.size()):
                                item = part.get(z).asList()
                                face_part = [item.get(0).asDouble(), item.get(1).asDouble(), item.get(2).asDouble()]
                                face_person.append(face_part)
                        else:
                            body_part = [part.get(1).asDouble(), part.get(2).asDouble(), part.get(3).asDouble()]
			    body_person.append(body_part)

                if body_person and face_person:
                    body.append(body_person)
                    face.append(face_person)

    poses, conf_poses = load_many_poses(body)
    faces, conf_faces = load_many_faces(face)

    return poses, conf_poses, faces, conf_faces


# read openpose data from json file
def read_openpose_from_json(json_filename):

    with open(json_filename) as data_file:
        loaded = json.load(data_file)

        poses = []
        conf_poses = []
        faces = []
        conf_faces = []

        for arr in loaded["people"]:
            conf_poses.append(arr["pose_keypoints_2d"][2::3])
            arr_poses = np.delete(arr["pose_keypoints_2d"], slice(2, None, 3))
            poses.append(list(zip(arr_poses[::2], arr_poses[1::2])))

            conf_faces.append(arr["face_keypoints_2d"][2::3])
            arr_faces = np.delete(arr["face_keypoints_2d"], slice(2, None, 3))  # remove confidence values from the array
            faces.append(list(zip(arr_faces[::2], arr_faces[1::2])))

    return poses, conf_poses, faces, conf_faces


# compute features for all people in the image, for each detected subject returns (centroid, features vector)
def get_features(poses, conf_poses, faces, conf_faces):
    data = []

    for itP in range(0, len(poses)):
        try:
            # compute facial keypoints coordinates w.r.t. to head centroid
            features, centr = compute_head_face_features(poses[itP], conf_poses[itP], faces[itP], conf_faces[itP])
            # if minimal amount of facial keypoints was detected
            if features is not None:
                featMap = np.asarray(features)
                centr = np.asarray(centr)
                poseFeats = np.concatenate((centr, featMap))

                data.append(poseFeats)
        except Exception as e:
            print("Got Exception: " + str(e))

    return data

