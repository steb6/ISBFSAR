import os
import pickle
import torch.utils.data as data
import random
import numpy as np


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

        if self.return_true_class_index:  # To print the name of the action
            return support_set, target_set, np.array(support_classes), target_class
        else:  # To train the model
            return support_set, target_set, np.array(range(0, self.k)), np.array(support_classes.index(target_class))

    def __len__(self):
        return self.n_task
