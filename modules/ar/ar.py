import pickle
from collections import OrderedDict
import numpy as np
from modules.ar.utils.model import TRXOS
from utils.params import TRXConfig
import torch
from tqdm import tqdm


class ActionRecognizer:
    def __init__(self, args, add_hook=False):
        self.input_type = args.input_type
        self.device = args.device

        self.ar = TRXOS(TRXConfig(), add_hook=add_hook)
        self.ar.load_state_dict(torch.load(args.final_ckpt_path,
                                           map_location=torch.device(0))['model_state_dict'])
        self.ar.cuda()
        self.ar.eval()

        self.support_set = OrderedDict()
        self.previous_frames = []
        self.seq_len = args.seq_len
        self.way = args.way
        self.n_joints = args.n_joints if args.input_type == "skeleton" else 0

        self.requires_focus = [False for _ in range(args.way)]

    def inference(self, data):
        """
        It receives an iterable of data that contains poses, images or both
        """
        if data is None:
            return {}, 0

        if len(self.support_set) == 0:  # no class to predict
            return {}, 0

        # Process new frame
        data = [torch.FloatTensor(x).cuda() for x in data]
        self.previous_frames.append(data)
        if len(self.previous_frames) < self.seq_len:  # few samples
            return {}, 0
        elif len(self.previous_frames) == self.seq_len + 1:
            self.previous_frames = self.previous_frames[1:]  # add as last frame

        # Predict actual action
        data = [torch.stack([i[j] for i in self.previous_frames]) for j in range(len(self.previous_frames[0]))]
        labels = torch.IntTensor(list(range(self.way))).unsqueeze(0).cuda()
        ss = []
        if self.input_type in ["skeleton", "hybrid"]:
            ss.append(torch.stack([self.support_set[c]["poses"] for c in self.support_set.keys()]))
            ss = [x.reshape(*x.shape[:-2], -1).unsqueeze(0) for x in ss]
            data[0] = data[0].reshape(*data[0].shape[:-2], -1).unsqueeze(0)  # Add batch dimension, data is a list
        if self.input_type in ["rgb", "hybrid"]:
            ss.append(torch.stack([self.support_set[c]["imgs"] for c in self.support_set.keys()]).unsqueeze(0))
            data[0] = data[0].unsqueeze(0)
        pad = torch.zeros_like(ss[0])
        while ss[0].shape[1] < 5:
            ss[0] = torch.concat((ss[0], pad), dim=1)
        outputs = self.ar(ss, labels, data)  # TODO INFERENCE SHOULD NOT COMPUTE PROTOTYPES EVERY TIME

        # Softmax
        few_shot_result = torch.softmax(outputs['logits'].squeeze(0), dim=0).detach().cpu().numpy()
        open_set_result = outputs['is_true'].squeeze(0).detach().cpu().numpy()

        # Return output
        results = {}
        for k in range(len(self.support_set)):
            results[list(self.support_set.keys())[k]] = (few_shot_result[k])
        return results, open_set_result

    def remove(self, flag):
        self.support_set.pop(flag)

    def train(self, inp):
        self.support_set[inp['flag']] = {c: torch.FloatTensor(inp['data'][c]).cuda() for c in inp['data'].keys()}


if __name__ == "__main__":
    ar = ActionRecognizer(TRXConfig())
    for _ in range(5):
        ar.train((np.random.rand(16, 90), "test{}".format(_)))
    for _ in tqdm(range(100000)):
        ar.inference(np.random.rand(90))
