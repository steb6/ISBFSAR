import json
import os
import pickle
import torch.utils.data as data
import random
import numpy as np


class MetrabsData(data.Dataset):
    def __init__(self, path, k=5, skeleton='smpl+head_30', mode='train', n_task=10000, train_size=0.8,
                 debug_classes=False):
        self.n_task = n_task
        self.path = path
        self.k = k
        self.classes = next(os.walk(self.path))[1]  # Get list of directories

        for elem in self.classes:
            if 'other_person' in elem:
                self.classes.remove(elem)
        self.classes.remove('handshaking')

        self.debug_classes = debug_classes

        n = int(len(self.classes) * train_size)
        if mode == 'train':
            self.classes = self.classes[:n]
        if mode == 'valid':
            self.classes = self.classes[n:]

        self.n_classes = len(self.classes)
        with open('assets/skeleton_types.pkl', "rb") as input_file:
            self.skeleton_types = pickle.load(input_file)
        self.skeleton = skeleton

    def get_random_video(self, id):
        sequences = next(os.walk(os.path.join(self.path, self.classes[id])))[2]
        path = random.sample(sequences, 1)[0]
        # path = sequences[0]  # TODO REMOVE DEBUG
        path = os.path.join(self.path, self.classes[id], path)
        with open(path, 'rb') as file:
            elem = json.load(file)
        elem = np.array(elem)[:, self.skeleton_types[self.skeleton]['indices'], :]
        return elem

    def __getitem__(self, idx):  # Must return complete, imp_x and impl_y
        # support_classes = [0, 1, 2, 3, 4]  # TODO REMOVE DEBUG
        # target_class = np.array(2)  # TODO REMOVE DEBUG
        support_classes = random.sample(range(0, self.n_classes), self.k)
        target_class = np.array(random.sample(support_classes, 1)[0])

        support_set = np.array([self.get_random_video(cl) for cl in support_classes])/2200.  # TODO PARAMETRIZE
        target_set = self.get_random_video(target_class)/2200.  # TODO PARAMETRIZE

        if self.debug_classes:
            return support_set, target_set, np.array(support_classes), np.array(target_class)
        else:
            return support_set, target_set, np.array(range(0, self.k)), np.array(support_classes.index(target_class))

    def __len__(self):
        return self.n_task
