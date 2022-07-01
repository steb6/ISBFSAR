import pickle
import numpy as np
from modules.ar.utils.model import Skeleton_TRX_Disc as CNN_TRX
from utils.params import TRXConfig
# from utils.tensorrt_runner import Runner
import torch
from tqdm import tqdm
import copy


class ActionRecognizer:
    def __init__(self, args):
        self.device = args.device

        # self.ar = Runner(args.trt_path)
        self.ar = CNN_TRX(TRXConfig())
        self.ar.load_state_dict(torch.load(args.final_ckpt_path,
                                           map_location=torch.device(0))['model_state_dict'])
        self.ar.cuda()
        self.ar.eval()

        self.support_set = torch.zeros((args.way, args.seq_len, args.n_joints * 3)).float()
        self.previous_frames = []
        self.support_labels = []
        self.seq_len = args.seq_len
        self.way = args.way
        self.n_joints = args.n_joints
        self.similar_actions = []

        self.requires_focus = [False for _ in range(args.way)]
        self.requires_box = [None for _ in range(args.way)]

    def inference(self, pose):
        if pose is None:
            return {}, 0

        if len(self.support_labels) == 0:  # no class to predict
            return {}, 0

        pose = torch.FloatTensor(pose.reshape(-1)).cuda()

        self.previous_frames.append(pose)
        if len(self.previous_frames) < self.seq_len:  # few samples
            return {}, 0
        elif len(self.previous_frames) == self.seq_len + 1:
            self.previous_frames = self.previous_frames[1:]  # add as last frame

        # Predict actual action
        poses = torch.stack(self.previous_frames).unsqueeze(0).cuda()
        labels = torch.IntTensor(list(range(self.way))).unsqueeze(0).cuda()
        with torch.no_grad():
            ss = self.support_set.unsqueeze(0).cuda()
        outputs = self.ar(ss, labels, poses)

        # Softmax
        few_shot_result = torch.softmax(outputs['logits'].squeeze(0), dim=0).detach().cpu().numpy()
        open_set_result = outputs['is_true'].squeeze(0).detach().cpu().numpy()

        results = {}  # return output as dictionary
        predicted = few_shot_result[:len(self.support_labels)]
        for k in range(len(predicted)):
            if k < len(self.support_labels):
                results[self.support_labels[k]] = (predicted[k])
            else:
                results['Action_{}'.format(k)] = (predicted[k])
        return results, open_set_result

    def sim(self, action1, action2):
        self.similar_actions.append((action1, action2))

    def remove(self, flag):
        """
        flag: Str
        """
        index = self.support_labels.index(flag)
        self.support_labels.remove(flag)
        self.support_set[index] = torch.zeros_like(self.support_set[index])
        self.requires_focus[index] = False

    def debug(self):
        with open('assets/skeleton_types.pkl', "rb") as inp:
            sk = pickle.load(inp)
        ed = sk['smpl+head_30']['edges']
        for i in range(len(self.support_set)):
            label = 'None' if i >= len(self.support_labels) else self.support_labels[i]
            yield self.support_set[i], label, ed

    def train(self, raw):
        """
        raw: Tuple ( FloatTensor Nx30x3, Str)
        """
        if raw is not None:  # if some data are given
            # Convert raw
            x = torch.FloatTensor(raw[0].reshape(self.seq_len, -1)).cuda()
            if raw[1] not in self.support_labels and len(self.support_labels) < 5:
                self.support_labels.append(raw[1])
            y = int(self.support_labels.index(raw[1]))
            self.support_set[y] = x


if __name__ == "__main__":
    ar = ActionRecognizer(TRXConfig())
    for _ in range(5):
        ar.train((np.random.rand(16, 90), "test{}".format(_)))
    for _ in tqdm(range(100000)):
        ar.inference(np.random.rand(90))
