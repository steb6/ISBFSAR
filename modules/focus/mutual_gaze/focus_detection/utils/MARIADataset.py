import os
import torch.utils.data as data
from modules.focus.mutual_gaze.focus_detection.utils.augmentations import *
import random
from utils.params import MutualGazeConfig
from tqdm import tqdm


class MARIAData(data.Dataset):
    def __init__(self, path, mode="train", augmentation_size=0.4, valid_size=0.2, split_number=0, model="facenet"):
        # Augmentation_size < 0 means always augment
        self.model = model
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
        self.images, self.labels = zip(*aux)
        self.images = list(self.images)
        self.labels = list(self.labels)

        # Balance data
        self.n_watch = sum(self.labels)
        self.n_not_watch = len(self.labels) - sum(self.labels)
        i = 0
        if self.n_watch > self.n_not_watch:
            for _ in range(self.n_watch - self.n_not_watch):
                is_watching = True
                while is_watching:
                    is_watching = self.labels[i]
                    if not is_watching:
                        self.images.append(self.images[i])
                        self.labels.append(self.labels[i])
                    i += 1
        else:
            for _ in range(self.n_not_watch - self.n_watch):
                is_watching = False
                while not is_watching:
                    is_watching = self.labels[i]
                    if is_watching:
                        self.images.append(self.images[i])
                        self.labels.append(self.labels[i])
                    i += 1
        aux = list(zip(self.images, self.labels))
        random.shuffle(aux)
        # self.images, self.labels = zip(*aux)

        # Augmentation
        self.augmentation_size = augmentation_size
        if self.mode == "train" and augmentation_size > 0:
            self.augment = [False for _ in range(len(aux))]
            aux = aux + aux[:int(len(aux) * augmentation_size)]
            self.augment = self.augment + [True for _ in range(len(aux) - len(self.augment))]
            self.aug_values = [[random.uniform(-0.2, 0.2),
                                random.uniform(-0.2, 0.2),
                                random.uniform(0.5, 2),
                                random.uniform(0.9, 1),
                                random.random() < 0.5,
                                int(random.uniform(-30, 30))] for _ in range(len(self.augment))]
        aux = list(zip(self.images, self.labels))
        random.shuffle(aux)
        self.images, self.labels = zip(*aux)

        self.n_watch = sum(self.labels)
        self.n_not_watch = len(self.labels) - sum(self.labels)

        self.sessions = sessions

    def __getitem__(self, idx):
        img = cv2.imread(self.images[idx])
        img = cv2.resize(img, (224, 224))
        label = self.labels[idx]

        if self.mode == "train":
            if self.augmentation_size > 0:  # Maria
                if self.augment[idx]:
                    img = horizontal_shift(img, value=self.aug_values[idx][0])
                    img = vertical_shift(img, value=self.aug_values[idx][1])
                    img = brightness(img, value=self.aug_values[idx][2])
                    img = zoom(img, value=self.aug_values[idx][3])
                    if self.aug_values[idx][4]:
                        img = horizontal_flip(img, False)
                    # img = rotation(img, value=self.aug_values[idx][5])
            else:  # Mine
                img = horizontal_shift(img, 0.2)
                img = vertical_shift(img, 0.2)
                img = brightness(img, 0.5, 2)
                img = zoom(img, 0.9)
                if random.random() < 0.5:
                    img = horizontal_flip(img, False)
                img = rotation(img, 30)

        if True:
            mean = [0.485, 0.456, 0.406]  # mean for resnet
            std = [0.229, 0.224, 0.225]  # std for resnet
            # # ZOO
            # if self.model == "resnet":
            #     mean = [0.485, 0.456, 0.406]  # mean for resnet
            #     std = [0.229, 0.224, 0.225]  # std for resnet
            # elif self.model == "resnet":
            #     mean = [0.28144035, 0.2883833,  0.33656912]  # mean for resnet
            #     std = [0.24786218, 0.23126106, 0.23487674]  # std for resnet
            img = img / 255.
            img = img - np.array(mean)
            img = img / np.array(std)

        return (img, label), label

    def __len__(self):
        return len(self.images)


class ComputeDatasetStatistics(data.Dataset):
    def __init__(self, path):
        # Augmentation_size < 0 means always augment
        sessions = os.listdir(path)
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
        self.images, self.labels = zip(*aux)
        # Count classes
        self.n_watch = sum(self.labels)
        self.n_not_watch = len(self.labels) - sum(self.labels)
        self.samples = []

    def __getitem__(self, idx):
        img = cv2.imread(self.images[idx])
        label = self.labels[idx]

        img = img / 255
        img = img - np.array([0.28144035, 0.2883833,  0.33656912])
        img = img / np.array([0.24786218, 0.23126106, 0.23487674])
        self.samples.append(img)

        return (img, label), label

    def __len__(self):
        return len(self.labels)

    def print_statistics(self):
        print(np.array(self.samples).mean(axis=(0, 1, 2)))
        print(np.array(self.samples).std(axis=(0, 1, 2)))
        print(len(self.samples))


if __name__ == "__main__":
    config = MutualGazeConfig()

    # stats = ComputeDatasetStatistics("D:/datasets/useless/"+config.dataset)
    # for elem in tqdm(stats):
    #     pass
    # stats.print_statistics()

    data = MARIAData("D:/datasets/useless/focus_dataset_heads", mode="train",
                     augmentation_size=config.augmentation_size)
    print(data.n_watch)
    print(data.n_not_watch)
    print(len(data))

    for (i, _), l in data:
        print(l)

        winname = "Test"
        cv2.namedWindow(winname)  # Create a named window
        cv2.moveWindow(winname, 40, 30)  # Move it to (40,30)
        cv2.imshow(winname, i)
        cv2.waitKey()
        cv2.destroyAllWindows()
