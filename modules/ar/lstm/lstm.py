import torch
import torch.nn as nn
import time
from torch.nn import LSTM
import copy
import json
from utils.misc import th_delete, oneHotVectorize, two_poses_movement, two_poses_movement_torch
import os
from scipy.spatial.transform import Rotation as R
import random


class Net(nn.Module):
    def __init__(self, hidden_size, seq_len, n_joints, n_classes=1):
        super(Net, self).__init__()
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(n_joints*3, hidden_size)
        self.lstm = LSTM(hidden_size, hidden_size)
        self.drop1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(hidden_size * seq_len, n_classes)
        self.drop2 = nn.Dropout(p=0.5)
        self.act_out = torch.nn.Sigmoid()
        self.weights = None

    def forward(self, h):
        h = self.fc1(h)
        h = self.drop1(h)
        h = self.lstm(h)[0]
        h = self.drop2(h)
        h = h.reshape(h.size(0), -1)
        h = self.fc2(h)
        return self.act_out(h)

    def reset_weights(self):
        torch.nn.init.xavier_uniform_(self.lstm.all_weights[0][0])
        torch.nn.init.xavier_uniform_(self.lstm.all_weights[0][1])
        torch.nn.init.zeros_(self.lstm.all_weights[0][2])
        torch.nn.init.zeros_(self.lstm.all_weights[0][3])

        torch.nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.zeros_(self.fc2.bias)


class ActionRecognizer:
    def __init__(self, args):

        self.device = args.device
        self.model = Net(args.hidden_size, args.seq_len, args.n_joints).to(args.device)
        self.optimizer = torch.optim.Adam(params=self.model.parameters())
        self.loss = torch.nn.BCELoss(reduction="sum")

        self.seq_len = args.seq_len
        self.n_joints = args.n_joints
        self.n_epochs = args.n_epochs
        self.augmentation_factor = args.augmentation_factor
        self.validation_size = args.validation_size
        self.augmentation_noise = args.augmentation_noise
        self.patience = args.patience

        self.dataset = []  # declare necessary parameters
        self.previous = []
        self.classes = []

    def inference(self, pose):
        """
        pose: FloatTensor 30x3 already normalized
        """
        if pose is None:
            return None

        if len(self.classes) == 0:  # no class to predict
            return None

        pose = torch.FloatTensor(pose)  # move tensor
        pose = pose.to(self.device)

        if len(self.previous) < self.seq_len:  # few samples
            self.previous.append(pose)
            return None

        self.previous = self.previous[1:self.seq_len]  # add as last frame
        self.previous.append(pose)

        # Predict actual action
        poses = torch.stack(self.previous).reshape(self.seq_len, -1).to(self.device)
        with torch.no_grad():
            output = self.model(poses.unsqueeze(0))  # add batch dimension and give to model

        results = {}  # return output as dictionary
        for k in range(len(output[0])):
            results[self.classes[k]] = output[0][k].item()
        return results

    def remove(self, flag):  # TODO fix this
        """
        flag: Str
        """
        new_dataset = []
        for elem in self.dataset:
            if self.classes[elem[1].max(0)[1]] == flag:  # Remove line of dataset with y equal to to_erase
                continue
            reduced = th_delete(elem[1], self.classes.index(flag))
            new_dataset.append((elem[0], reduced))
        self.classes.remove(flag)
        self.dataset = new_dataset
        self.train(None)

    def debug_dataset(self):
        """
        It saves self.dataset and self.class inside a 'data' folder, such that they can be seen with debug-data.py
        """
        print("Dataset has length ", len(self.dataset))
        for i in range(len(self.classes)):
            count = 0
            for x, y in self.dataset:
                if y.max(0)[1] == i:
                    count += 1
            print("Class ", self.classes[i], " has ", int(count), " elements ", "{:.2f}".format(count / len(self.dataset)))

        # Save dataset as txt
        if not os.path.exists('data'):
            os.mkdir('data')
        dataset_list = [(xa.tolist(), ya.tolist()) for xa, ya in self.dataset]
        with open('data' + os.sep + 'dataset.txt', 'w') as outfile:
            json.dump(dataset_list, outfile)
        with open('data' + os.sep + 'classes.txt', 'w') as outfile:
            json.dump(self.classes, outfile)

    def prepare_and_add_data(self, x, y, rotate=True, symmetry=True):

        # Trim x (extract all subsequences if possible)
        trimmed_x = []
        trimmed_y = []
        if len(x) == self.seq_len:
            trimmed_x.append(x)
            trimmed_y.append(y)
        else:
            for i in range(len(x) - self.seq_len):
                trimmed_x.append(x[i:self.seq_len + i])
                trimmed_y.append(y)
        x = torch.stack(trimmed_x)
        y = torch.stack(trimmed_y)

        # TODO KEEP ONLY SUBSEQUENCES WITH MOVEMENT
        # 14 8 30 3
        # movements = []
        # for window in x:
        #     m = 0
        #     for i in range(params["window_size"] - 1):
        #         m += two_poses_movement_torch(window[i], window[i+1])
        #     movements.append(m)
        # movements = torch.stack(movements)
        # mean = torch.mean(movements)
        # print(mean)
        # print("Kept {} windows over a total of {} windows".format(torch.sum(movements > mean), len(x)))
        # x = x[movements > mean]
        # y = y[movements > mean]
        # TODO END

        # Data augmentation
        augmented_x = []
        augmented_y = []
        for xa, ya in zip(x, y):
            for _ in range(self.augmentation_factor):
                cop = copy.deepcopy(xa)

                # TODO MAKE SLOWER
                # TODO MAKE FASTER

                cop = cop + torch.randn_like(cop) * self.augmentation_noise
                cop = torch.clamp(cop, min=-1, max=1)

                if rotate:
                    angle = random.uniform(0, 360)
                    rot = R.from_euler('xyz', (0, angle, 0)).as_matrix()
                    rot = torch.FloatTensor(rot).to(self.device)
                    cop = torch.matmul(cop, rot)

                if symmetry:
                    if random.random() < 0.5:
                        cop[..., 0] = -cop[..., 0]

                augmented_x.append(cop)
                augmented_y.append(ya)

        x = torch.stack(augmented_x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        y = torch.stack(augmented_y)

        # Reshape and modify ys
        y = y.view(y.size(0), -1)
        y = oneHotVectorize(y, size=len(self.classes))
        y = y.float()

        # Add preprocessed data to dataset
        assert len(x) == len(y)
        new_dataset = []
        for elem in self.dataset:
            trg = elem[1]
            while len(trg) < len(self.classes):  # append 0 to old yi to get right dimension
                trg = torch.cat((trg, torch.zeros(1).to(self.device)), dim=0)
            new_dataset.append((elem[0], trg))  # append to new dataset as tuple

        self.dataset = new_dataset

        for i in range(len(x)):
            self.dataset.append((x[i], y[i]))

        random.shuffle(self.dataset)  # the shuffle is done in place, it returns None

    def train(self, raw, rot=True, sim=True):
        """
        raw: Tuple ( FloatTensor Nx30x3, Str)
        """
        if raw is not None:  # if some data are given
            # Convert raw
            x = torch.FloatTensor(raw[0]).to(self.device)
            if raw[1] not in self.classes:
                self.classes.append(raw[1])
            y = torch.IntTensor([int(self.classes.index(raw[1]))]).to(self.device)
            self.prepare_and_add_data(x, y, rot, sim)  # TODO add rotation augmentation

        # self.debug_dataset()  # TODO REMOVE

        # Create validation set
        validation = self.dataset[:int(len(self.dataset) * self.validation_size)]
        training = self.dataset[int(len(self.dataset) * self.validation_size):]

        print("Training size: ", len(training), ", validation size:", len(validation))

        # Restart model
        self.model = Net(self.model.hidden_size, self.seq_len, self.n_joints,
                         n_classes=len(self.classes)).to(self.device)
        self.model.train()
        self.optimizer = torch.optim.Adam(params=self.model.parameters())

        if len(self.classes) == 0:  # Nothing to do
            return

        #################
        # TRAINING loop #
        #################

        # Initialize loop variables
        min_val_loss = 9999
        no_improvement = 0
        init = time.time()

        for epoch in range(self.n_epochs):

            train_loss = 0
            train_accuracy = 0
            val_loss = 0
            val_accuracy = 0
            random.shuffle(training)

            self.model.train()  # TRAINING
            for x, y in training:

                x = x.unsqueeze(0)  # add batch dimension
                y = y.unsqueeze(0)  # add batch dimension

                self.model.zero_grad()  # Compute output
                output = self.model(x)
                true = y.max(1)[1]
                out = output.max(1)[1]
                if true == out:
                    train_accuracy += 1

                ls = self.loss(output, y)  # Compute loss
                ls = ls  # * (len(training) + len(validation)) / y_counter[true]  # TODO CHECK #########################
                train_loss += ls.item()

                ls.backward()
                self.optimizer.step()

            self.model.eval()  # VALIDATION
            with torch.no_grad():
                for x, y in validation:

                    x = x.unsqueeze(0)  # add batch dimension
                    y = y.unsqueeze(0)  # add batch dimension

                    output = self.model(x)  # Compute output
                    true = y.max(1)[1]
                    out = output.max(1)[1]
                    if true == out:
                        val_accuracy += 1

                    ls = self.loss(output, y)  # Compute loss
                    ls = ls  # * (len(training) + len(validation))  # / y_counter[true]  # TODO CHECK ##################
                    val_loss += ls.item()

            train_loss = train_loss / len(training)
            train_accuracy = train_accuracy / len(training)
            val_loss = val_loss / len(validation)
            val_accuracy = val_accuracy / len(validation)

            print("Epoch {}, train_loss: {:.4f}, train_accuracy: {:.4f}, validation_loss: {:.4f}, "
                  "validation_accuracy: {:.4f}".format(epoch, train_loss, train_accuracy, val_loss, val_accuracy))

            # Early stopping
            if val_loss < min_val_loss:
                min_val_loss = val_loss
                no_improvement = 0
            else:
                no_improvement += 1
            if no_improvement > self.patience:
                print('Stop training because no improvements')
                break
            # if train_loss < 0.1:
            #     print("Stop training: low training loss")
            #     break
            if val_accuracy == 1.0:
                print("Maximum validation accuracy reached")
                break

        ending = time.time()
        print('training time: {:.2f}sec'.format(ending - init))


if __name__ == "__main__":
    ar = ActionRecognizer()
    while True:
        ar.train((torch.rand(20, 30, 3), "test"))
        for _ in range(20):
            ar.inference(torch.rand(30, 3))
