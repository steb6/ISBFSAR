import os
import cv2
import torch.utils.data as data
import random

from modules.focus.mutual_gaze.focus_detection.augmentations import horizontal_shift, vertical_shift, brightness, zoom, \
    channel_shift, horizontal_flip, rotation


class MARIAData(data.Dataset):
    def __init__(self, path, mode="train", train_size=0.8):
        self.path = path
        sessions = os.listdir(path)
        self.mode = mode
        if mode == "train":
            sessions = sessions[:int(len(sessions) * train_size)]
        else:
            sessions = sessions[int(len(sessions) * train_size):]
        self.images = []
        self.labels = []
        for session in sessions:
            watching = os.listdir(os.path.join(path, session, 'watching'))
            for _ in range(2 if self.mode == "train" else 1):
                self.images += [os.path.join(path, session, 'watching', w) for w in watching]
            not_watching = os.listdir(os.path.join(path, session, 'not_watching'))
            self.images += [os.path.join(path, session, 'not_watching', n) for n in not_watching]
            for _ in range(2 if self.mode == "train" else 1):
                self.labels += [True for _ in range(len(os.listdir(os.path.join(path, session, 'watching'))))]
            self.labels += [False for _ in range(len(os.listdir(os.path.join(path, session, 'not_watching'))))]
        aux = list(zip(self.images, self.labels))
        random.shuffle(aux)
        self.images, self.labels = zip(*aux)
        self.n_watch = sum(self.labels)
        self.n_not_watch = len(self.labels) - sum(self.labels)

    def __getitem__(self, idx):
        img = cv2.imread(self.images[idx])
        label = self.labels[idx]

        # TODO MAKE AUGMENTATION LOOKS LIKE THE OTHERS
        if self.mode == "train":
            img = horizontal_shift(img, 0.2)
            img = vertical_shift(img, 0.2)
            img = brightness(img, 0.5, 2)
            img = zoom(img, 0.9)
            # img = channel_shift(img, 5)
            if random.random() < 0.5:
                img = horizontal_flip(img, False)
            img = rotation(img, 30)

        return img, label

    def __len__(self):
        return len(self.labels)


if __name__ == "__main__":
    data = MARIAData("D:/datasets/focus_dataset")
    for elem in data:
        print(elem[1])
        cv2.imshow("", elem[0])
        cv2.waitKey(0)
