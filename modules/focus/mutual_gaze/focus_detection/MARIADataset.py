import os
import cv2
import torch.utils.data as data
import random


class MARIAData(data.Dataset):
    def __init__(self, path, mode="train", train_size=0.8):
        self.path = path
        sessions = os.listdir(path)
        if mode == "train":
            sessions = sessions[:int(len(sessions) * train_size)]
        else:
            sessions = sessions[int(len(sessions) * train_size):]
        self.images = []
        self.labels = []
        for session in sessions:
            watching = os.listdir(os.path.join(path, session, 'watching'))
            self.images += [os.path.join(path, session, 'watching', w) for w in watching]
            not_watching = os.listdir(os.path.join(path, session, 'not_watching'))
            self.images += [os.path.join(path, session, 'not_watching', n) for n in not_watching]
            self.labels += [True for _ in range(len(os.listdir(os.path.join(path, session, 'watching'))))]
            self.labels += [False for _ in range(len(os.listdir(os.path.join(path, session, 'not_watching'))))]
        aux = list(zip(self.images, self.labels))
        random.shuffle(aux)
        self.images, self.labels = zip(*aux)

    def __getitem__(self, idx):
        img = cv2.imread(self.images[idx])
        label = self.labels[idx]
        return img, label

    def __len__(self):
        return len(self.labels)


if __name__ == "__main__":
    data = MARIAData("D:/datasets/focus_dataset")
    for elem in data:
        print(elem[1])
        cv2.imshow("", elem[0])
        cv2.waitKey(0)
