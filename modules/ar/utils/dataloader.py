import os
import pickle
import torch.utils.data as data
import random
import numpy as np


# https://rose1.ntu.edu.sg/dataset/actionRecognition/


class MetrabsData(data.Dataset):
    def __init__(self, path, k=5, n_task=10000, return_true_class_index=False):
        self.n_task = n_task
        self.path = path
        self.k = k
        self.classes = next(os.walk(self.path))[1]  # Get list of directories
        self.return_true_class_index = return_true_class_index

    def get_random_video(self, id):
        sequences = next(os.walk(os.path.join(self.path, self.classes[id])))[2]
        path = random.sample(sequences, 1)[0]
        path = os.path.join(self.path, self.classes[id], path)
        with open(path, 'rb') as file:
            elem = pickle.load(file)
        return elem

    def __getitem__(self, idx):  # Must return complete, imp_x and impl_y
        support_classes = random.sample(range(0, len(self.classes)), self.k)
        target_class = np.array(random.sample(support_classes, 1)[0])

        support_set = np.array([self.get_random_video(cl) for cl in support_classes])
        target_set = self.get_random_video(target_class)
        unknown_set = self.get_random_video(random.sample(list(np.delete(np.array(list(range(len(self.classes)))),
                                                                         support_classes)), 1)[0])

        if self.return_true_class_index:  # To print the name of the action
            return support_set, target_set, np.array(support_classes), target_class
        else:  # To train the model
            return support_set, target_set, unknown_set,\
                   np.array(range(0, self.k)), np.array(support_classes.index(target_class))

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
            with open(os.path.join(example,'0.pkl'), 'rb') as file:
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


