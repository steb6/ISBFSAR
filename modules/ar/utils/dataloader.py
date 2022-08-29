import os
import pickle
import torch.utils.data as data
import random
import numpy as np
from torchvision.transforms import transforms
import cv2
import torch


# https://rose1.ntu.edu.sg/dataset/actionRecognition/


class EpisodicLoader(data.Dataset):
    def __init__(self, path, k=5, n_task=10000, input_type="skeleton"):
        self.n_task = n_task
        self.path = path
        self.k = k
        self.classes = next(os.walk(self.path))[1]  # Get list of directories
        self.input_type = input_type
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def get_random_sample(self, class_name):
        sequences = next(os.walk(os.path.join(self.path, class_name)))[2 if self.input_type == "skeleton" else 1]
        # path = None
        # SOME FOLDERS APPEARS TO BE EMPTY, IN THE CASE DO RESAMPLING
        # while path is None or len(os.listdir(path)) == 0:
        path = random.sample(sequences, 1)[0]  # Get random file
        path = os.path.join(self.path, class_name, path)  # Create full path
        sample = None

        if self.input_type == "skeleton":
            with open(path, 'rb') as file:
                sample = pickle.load(file)
        if self.input_type == "rgb":
            imgs = []
            for img_number in range(8):
                img = cv2.imread(os.path.join(path, f"{img_number}.png"))
                img = img / 255.  # TODO READD
                img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])  # TODO READD
                imgs.append(img)
            sample = np.array(imgs)
            # sample = sample[list(range(0, 16, 2))]  # TODO IT RETURNS 8 IMAGES INSTEAD OF 16
        return sample

    def __getitem__(self, _):  # Must return complete, imp_x and impl_y
        support_classes = random.sample(self.classes, self.k)
        target_class = random.sample(support_classes, 1)[0]
        unknown_class = random.sample([x for x in self.classes if x not in support_classes], 1)[0]

        support_set = [self.get_random_sample(cl) for cl in support_classes]
        target_set = self.get_random_sample(target_class)
        unknown_set = self.get_random_sample(unknown_class)

        support_set = np.array(support_set)
        target_set = np.array(target_set)
        unknown_set = np.array(unknown_set)

        # # TODO REMOVE DEBUG
        # print(support_classes)
        # for a in support_set:
        #     for b in a:
        #         cv2.imshow("", b)
        #         cv2.waitKey(0)
        # print(target_class)
        # for a in target_set:
        #     cv2.imshow("", a)
        #     cv2.waitKey(0)
        # print(unknown_class)
        # for a in unknown_set:
        #     cv2.imshow("", a)
        #     cv2.waitKey(0)
        # # TODO END REMOVE DEBUG

        return {'support_set': np.array(support_set),
                'target_set': np.array(target_set),
                'unknown_set': np.array(unknown_set),
                'support_classes': support_classes,
                'target_class': target_class,
                'unknown_class': unknown_class}

    def __len__(self):
        return self.n_task


class TestMetrabsData(data.Dataset):
    # Test classes: classes to add in support set
    # Os classes: other class to consider (together with test classes)

    def __init__(self, samples_path, exemplars_path, test_classes, os_classes):
        self.exemplars_classes = test_classes  # next(os.walk(exemplars_path))[1]
        exemplars_files = [os.path.join(exemplars_path, elem) for elem in self.exemplars_classes]
        self.exemplars_poses = []
        for example in exemplars_files:
            with open(os.path.join(example, '0.pkl'), 'rb') as file:
                elem = pickle.load(file)
            self.exemplars_poses.append(elem)
        self.exemplars_poses = np.stack(self.exemplars_poses)
        self.target_set = []
        self.target_classes = []
        self.target_names = []
        self.support_names = test_classes
        self.unknowns = []
        self.unknowns_names = []

        for c in self.exemplars_classes:
            class_path = os.path.join(samples_path, c)
            files = next(os.walk(class_path))[2]
            files = [os.path.join(class_path, file) for file in files]
            self.target_set += files
            self.target_classes += [self.exemplars_classes.index(c) for _ in range(len(files))]
            self.target_names += [c for _ in range(len(files))]

        for c in os_classes:
            class_path = os.path.join(samples_path, c)
            files = next(os.walk(class_path))[2]
            files = [os.path.join(class_path, file) for file in files]
            self.target_set += files
            self.target_classes += [-1 for _ in range(len(files))]
            self.target_names += [c for _ in range(len(files))]

            self.unknowns += files
            self.unknowns_names += [c for _ in range(len(files))]

    def __getitem__(self, idx):  # Must return complete, imp_x and impl_y
        with open(self.target_set[idx], 'rb') as file:
            target_set = pickle.load(file)
        unknown = []
        unknown_name = []
        if len(self.unknowns) > 0:
            unknown_id = random.choice(list(range(len(self.unknowns))))
            unknown = self.unknowns[unknown_id]
            with open(unknown, 'rb') as file:
                unknown = pickle.load(file)
            unknown_name = self.unknowns_names[unknown_id]
        return self.exemplars_poses, target_set, unknown, \
               np.array(list(range(len(self.exemplars_classes)))), self.target_classes[idx], self.support_names, \
               self.target_names[idx], unknown_name

    def __len__(self):
        return len(self.target_set)
