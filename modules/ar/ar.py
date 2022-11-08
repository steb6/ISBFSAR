from collections import OrderedDict
import numpy as np
from ISBFSAR.modules.ar.utils.model import TRXOS
from ISBFSAR.utils.params import TRXConfig
import torch
from tqdm import tqdm
import copy


class ActionRecognizer:
    def __init__(self, args, add_hook=False):
        self.input_type = args.input_type
        self.device = args.device

        self.ar = TRXOS(TRXConfig(), add_hook=add_hook)
        # Fix dataparallel
        state_dict = torch.load(args.final_ckpt_path, map_location=torch.device(0))['model_state_dict']
        state_dict = OrderedDict({param.replace('.module', ''): data for param, data in state_dict.items()})
        self.ar.load_state_dict(state_dict)
        self.ar.cuda()
        self.ar.eval()

        self.support_set = OrderedDict()
        self.requires_focus = {}
        self.previous_frames = []
        self.seq_len = args.seq_len
        self.way = args.way
        self.n_joints = args.n_joints if args.input_type == "skeleton" else 0

    def inference(self, data):
        """
        It receives an iterable of data that contains poses, images or both
        """
        if data is None or len(data) == 0:
            return {}, 0, {}

        if len(self.support_set) == 0:  # no class to predict
            return {}, 0, {}

        # Process new frame
        data = {k: torch.FloatTensor(v).cuda() for k, v in data.items()}
        self.previous_frames.append(copy.deepcopy(data))
        if len(self.previous_frames) < self.seq_len:  # few samples
            return {}, 0, {}
        elif len(self.previous_frames) == self.seq_len + 1:
            self.previous_frames = self.previous_frames[1:]  # add as last frame

        # Prepare query with previous frames
        for t in list(data.keys()):
            data[t] = torch.stack([elem[t] for elem in self.previous_frames]).unsqueeze(0)
        labels = torch.IntTensor(list(range(len(self.support_set)))).unsqueeze(0).cuda()

        # Get SS
        ss = None
        ss_f = None
        if all('features' in self.support_set[c].keys() for c in self.support_set.keys()):
            ss_f = torch.stack([self.support_set[c]["features"] for c in self.support_set.keys()])  # 3 16 90
            pad = torch.zeros_like(ss_f[0]).unsqueeze(0)
            while ss_f.shape[0] < self.way:
                ss_f = torch.concat((ss_f, pad), dim=0)
            ss_f = ss_f.unsqueeze(0)  # Add batch dimension
        else:
            ss = {}
            if self.input_type in ["rgb", "hybrid"]:
                ss["rgb"] = torch.stack([self.support_set[c]["imgs"] for c in self.support_set.keys()]).unsqueeze(0)
            if self.input_type in ["skeleton", "hybrid"]:
                ss["sk"] = torch.stack([self.support_set[c]["poses"] for c in self.support_set.keys()]).unsqueeze(0)
        with torch.no_grad():
            outputs = self.ar(ss, labels, data, ss_features=ss_f)  # RGB, POSES

        # Save support features
        if ss_f is None:
            for i, s in enumerate(self.support_set.keys()):
                self.support_set[s]['features'] = outputs['support_features'][0][i]  # zero to remove batch dimension

        # Softmax
        few_shot_result = torch.softmax(outputs['logits'].squeeze(0), dim=0).detach().cpu().numpy()
        open_set_result = outputs['is_true'].squeeze(0).detach().cpu().numpy()

        # Return output
        results = {}
        for k in range(len(self.support_set)):
            results[list(self.support_set.keys())[k]] = (few_shot_result[k])
        return results, open_set_result, self.requires_focus

    def remove(self, flag):
        if flag in self.support_set.keys():
            self.support_set.pop(flag)
            self.requires_focus.pop(flag)
            return True
        else:
            return False

    def train(self, inp):
        self.support_set[inp['flag']] = {c: torch.FloatTensor(inp['data'][c]).cuda() for c in inp['data'].keys()}
        self.requires_focus[inp['flag']] = inp['requires_focus']


if __name__ == "__main__":
    ar = ActionRecognizer(TRXConfig())
    for _ in range(5):
        ar.train((np.random.rand(16, 90), "test{}".format(_)))
    for _ in tqdm(range(100000)):
        ar.inference(np.random.rand(90))
