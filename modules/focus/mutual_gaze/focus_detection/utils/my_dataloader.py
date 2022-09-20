import os
import torch.utils.data as data
from modules.focus.mutual_gaze.focus_detection.utils.augmentations import *
import json
import random
import cv2


class MARIAData(data.Dataset):
    def __init__(self, path, mode="train", valid_size=0.2, split_number=0):
        self.path = path
        self.mode = mode
        w = 'pxx_train' if mode != "test" else 'pxx_test'
        self.sessions = np.load(os.path.join(path, 'setsFile_participants.npz'))[w][split_number]
        if mode == "train":
            self.sessions = self.sessions[int(len(self.sessions) * valid_size):]
        elif mode == "valid":
            self.sessions = self.sessions[:int(len(self.sessions) * valid_size)]

        self.annotations = None
        with open(os.path.join(path, "realsense", "eyecontact_annotations.txt"), "r") as infile:
            self.annotations = infile.readlines()
        self.annotations = [x.split() for x in self.annotations]
        self.annotations = list(filter(lambda x: x[0].split('/')[1] in self.sessions, self.annotations))
        random.shuffle(self.annotations)

        self.n_watch = sum([int(x[1]) for x in self.annotations])
        self.n_not_watch = len(self.annotations) - sum([int(x[1]) for x in self.annotations])

    def __getitem__(self, idx):
        img = cv2.imread(os.path.join(self.path, "realsense", self.annotations[idx][0]))
        with open(os.path.join(self.path,
                               "realsense",
                               self.annotations[idx][0].replace("images_human",
                                                                "data_openpose").replace(".jpg",
                                                                                         "_keypoints.json"))) as infile:
            pose = np.array(json.load(infile)['people'][0]['face_keypoints_2d']).reshape(-1, 3).astype(np.float)

        img_ = img[int(np.min(pose[:, 1])):int(np.max(pose[:, 1])), int(np.min(pose[:, 0])):int(np.max(pose[:, 0]))]
        if img_.shape[0] > 0 and img_.shape[1] > 0:
            img_ = cv2.resize(img_, (224, 224))
        else:
            img_ = np.zeros((224, 224, 3))
        img_ = img_ / 255
        img_ = img_.swapaxes(-1, -2).swapaxes(-2, -3)
        pose_ = pose[[36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 68, 69]]
        if np.any(pose_):  # Sometimes they can be all zeros
            pose_ = pose_ - np.mean(pose_, axis=0)
            pose_ = pose_ / np.max(pose_)
        pose_ = pose_.reshape(-1).astype(np.float)

        return 0, pose, 0, pose_, int(self.annotations[idx][1])

    def __len__(self):
        return len(self.annotations)


if __name__ == "__main__":
    data = MARIAData("D:/datasets/mutualGaze_dataset", mode="train")

    for elem in data:
        frame, skeleton, _, p, _ = elem
        for point in skeleton:
            frame = cv2.circle(frame, (int(point[0]), int(point[1])), 1, (255, 0, 255))
        cv2.imshow("", frame)
        cv2.waitKey(0)
