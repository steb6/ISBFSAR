import os
import torch.utils.data as data
from modules.focus.mutual_gaze.focus_detection.utils.augmentations import *
import random


class MARIAData(data.Dataset):
    def __init__(self, path, mode="train", augmentation_size=0.2, valid_size=0.2, split_number=0):
        self.mode = mode
        if mode == "train":
            sessions = np.load('assets/setsFile_participants.npz')['pxx_train'][split_number]
            sessions = sessions[int(len(sessions) * valid_size):]
        elif mode == "test":
            sessions = np.load('assets/setsFile_participants.npz')['pxx_test'][split_number]
        elif mode == "valid":
            sessions = np.load('assets/setsFile_participants.npz')['pxx_train'][split_number]
            sessions = sessions[:int(len(sessions) * valid_size)]
        else:
            raise Exception("")
        self.images = []
        self.labels = []
        for session in sessions:
            session = str(session)
            watching = os.listdir(os.path.join(path, session, 'watching'))
            self.images += [os.path.join(path, session, 'watching', w) for w in watching]
            not_watching = os.listdir(os.path.join(path, session, 'not_watching'))
            self.images += [os.path.join(path, session, 'not_watching', n) for n in not_watching]
            self.labels += [True for _ in range(len(os.listdir(os.path.join(path, session, 'watching'))))]
            self.labels += [False for _ in range(len(os.listdir(os.path.join(path, session, 'not_watching'))))]
        aux = list(zip(self.images, self.labels))
        random.shuffle(aux)
        # Augmentation
        # TODO MARIA
        if self.mode == "train":
            self.augment = [False for _ in range(len(aux))]
            aux = aux + aux[:int(len(aux) * augmentation_size)]
            self.augment = self.augment + [True for _ in range(len(aux) - len(self.augment))]
            self.aug_values = [[random.uniform(-0.2, 0.2),
                                random.uniform(-0.2, 0.2),
                                random.uniform(0.5, 2),
                                random.uniform(0.9, 1),
                                random.random() < 0.5,
                                int(random.uniform(-30, 30))] for _ in range(len(self.augment))]
        # TODO END
        self.images, self.labels = zip(*aux)
        # Count classes
        self.n_watch = sum(self.labels)
        self.n_not_watch = len(self.labels) - sum(self.labels)

    def __getitem__(self, idx):
        img = cv2.imread(self.images[idx])
        label = self.labels[idx]

        if self.mode == "train":
            # TODO MARIA
            if self.augment[idx]:
                img = horizontal_shift(img, value=self.aug_values[idx][0])
                img = vertical_shift(img, value=self.aug_values[idx][1])
                img = brightness(img, value=self.aug_values[idx][2])
                img = zoom(img, value=self.aug_values[idx][3])
                if self.aug_values[idx][4]:
                    img = horizontal_flip(img, False)
                img = rotation(img, value=self.aug_values[idx][5])
            # TODO MINE
            # img = horizontal_shift(img, 0.2)
            # img = vertical_shift(img, 0.2)
            # img = brightness(img, 0.5, 2)
            # img = zoom(img, 0.9)
            # if random.random() < 0.5:
            #     img = horizontal_flip(img, False)
            # img = rotation(img, 30)
            # TODO END

        return img, label

    def __len__(self):
        return len(self.labels)


if __name__ == "__main__":
    import torch

    data = MARIAData("D:/datasets/focus_dataset_heads", mode="test")
    loader = torch.utils.data.DataLoader(data, batch_size=1, num_workers=2, shuffle=True)

    for elem in loader:
        print(elem[1])
        cv2.imshow("", elem[0][0].detach().numpy())
        cv2.waitKey(0)
