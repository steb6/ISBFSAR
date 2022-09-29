import os
import pickle
import torch.utils.data as data
import random
import numpy as np
import cv2
from utils.params import seq_len


# https://rose1.ntu.edu.sg/dataset/actionRecognition/


class EpisodicLoader(data.Dataset):
    def __init__(self, path, k=5, n_task=10000, l=16, input_type="hybrid"):
        self.path = path
        self.k = k
        self.n_task = n_task
        self.l = l
        self.input_type = input_type
        self.classes = next(os.walk(self.path))[1]  # Get list of directories

    def get_random_sample(self, class_name):
        sequences = next(os.walk(os.path.join(self.path, class_name)))[1]  # Just list of directories

        path = random.sample(sequences, 1)[0]  # Get first random file
        path = os.path.join(self.path, class_name, path)  # Create full path
        imgs = []
        poses = []
        i = 0
        while True:  # Load one image at time
            try:
                with open(os.path.join(path, f"{i}.pkl"), 'rb') as file:
                    # Load skeleton
                    pose = pickle.load(file)
                    poses.append(pose.reshape(-1))
                    # Load image
                    img = cv2.imread(os.path.join(path, f"{i}.png"))
                    img = cv2.resize(img, (224, 224))
                    img = img / 255.
                    img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                    imgs.append(img.swapaxes(-1, -2).swapaxes(-2, -3))
                    i += 1
                if len(poses) == self.l:
                    break
            except Exception as e:
                print("[ERROR]: Erase this directory:", path, e)
                # Reset
                path = random.sample(sequences, 1)[0]  # Get first random file
                path = os.path.join(self.path, class_name, path)  # Create full path
                imgs = []
                poses = []
                i = 0
        if seq_len == 8:
            return np.stack(imgs)[list(range(0, 16, 2))], np.stack(poses)[list(range(0, 16, 2))]
        return np.stack(imgs), np.stack(poses)

    def __getitem__(self, _):  # Must return complete, imp_x and impl_y
        support_classes = random.sample(self.classes, self.k)
        target_class = random.sample(support_classes, 1)[0]
        unknown_class = random.sample([x for x in self.classes if x not in support_classes], 1)[0]

        support_set = [self.get_random_sample(cl) for cl in support_classes]
        target_set = self.get_random_sample(target_class)
        unknown_set = self.get_random_sample(unknown_class)

        return {'support_set': {"rgb": np.stack([x[0] for x in support_set]),
                                "sk": np.stack([x[1] for x in support_set])},
                'target_set': {"rgb": target_set[0],
                               "sk": target_set[1]},
                'unknown_set': {"rgb": unknown_set[0],
                                "sk": unknown_set[1]},
                'support_classes': np.stack([self.classes.index(elem) for elem in support_classes]),
                'target_class': self.classes.index(target_class),
                'unknown_class': self.classes.index(unknown_class)}

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


if __name__ == "__main__":
    from utils.matplotlib_visualizer import MPLPosePrinter
    from utils.params import TRXConfig

    skeleton = 'smpl+head_30'
    with open('assets/skeleton_types.pkl', "rb") as input_file:
        skeleton_types = pickle.load(input_file)
    edges = skeleton_types[skeleton]['edges']

    loader = EpisodicLoader(TRXConfig().data_path, input_type="hybrid")
    vis = MPLPosePrinter()

    for asd in loader:
        sup = asd['support_set']
        trg = asd['target_set']
        unk = asd['unknown_set']

        print(asd['support_classes'])
        for c in range(5):
            for k in range(16):
                cv2.imshow("sup", sup[c][0][k])
                cv2.waitKey(1)

                vis.clear()
                vis.print_pose(sup[c][1][k], edges)
                vis.sleep(0.01)
