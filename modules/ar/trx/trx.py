import pickle
from multiprocessing import Queue
import numpy as np
import time
from modules.ar.trx.utils.model import CNN_TRX
import torch


class ActionRecognizer:
    def __init__(self, args, trt=False):
        self.device = args.device

        self.model = CNN_TRX(args).to(self.device)  # declare necessary machine learning things
        model_data = torch.load(args.model_path)
        self.model.load_state_dict(model_data['model_state_dict'])
        self.model.eval()

        self.support_set = torch.zeros(args.way, args.seq_len, args.n_joints*3).float().to(self.device)
        self.previous_frames = []
        self.support_labels = []
        self.seq_len = args.seq_len
        self.way = args.way
        self.n_joints = args.n_joints

    def inference(self, pose):
        """
        pose: FloatTensor 30x3 already normalized
        """
        if pose is None:
            return None

        if len(self.support_labels) == 0:  # no class to predict
            return None

        pose = pose.reshape(-1)
        pose = torch.FloatTensor(pose).to(self.device)

        self.previous_frames.append(pose)
        if len(self.previous_frames) < self.seq_len:  # few samples
            return None
        elif len(self.previous_frames) == self.seq_len + 1:
            self.previous_frames = self.previous_frames[1:]  # add as last frame

        # Predict actual action
        poses = torch.stack(self.previous_frames).reshape(self.seq_len, -1).to(self.device)
        with torch.no_grad():
            labels = list(range(len(self.support_labels)))
            labels = torch.FloatTensor(labels).to(self.device)
            ss = self.support_set.reshape(-1, 90)
            output = self.model(ss, labels, poses)  # add batch dimension and give to model

        # predicted = torch.argmax(output['logits'], dim=-1)
        predicted = torch.softmax(output['logits'], dim=-1)
        predicted = predicted.squeeze().detach().cpu().numpy()

        results = {}  # return output as dictionary
        for k in range(len(predicted)):
            if k < len(self.support_labels):
                results[self.support_labels[k]] = predicted[k]
            else:
                results['Action_{}'.format(k)] = predicted[k]
        return results

    def remove(self, flag):  # TODO fix this
        """
        flag: Str
        """
        index = self.support_labels.index(flag)
        self.support_labels.remove(flag)
        self.support_set[index] = torch.zeros_like(self.support_set[index])

    def debug(self):
        with open('assets/skeleton_types.pkl', "rb") as inp:
            sk = pickle.load(inp)
        ed = sk['smpl+head_30']['edges']
        for i in range(len(self.support_set)):
            label = 'None' if i >= len(self.support_labels) else self.support_labels[i]
            yield self.support_set.detach().cpu().numpy()[i], label, ed

    def train(self, raw):
        """
        raw: Tuple ( FloatTensor Nx30x3, Str)
        """
        if raw is not None:  # if some data are given
            # Convert raw
            x = torch.FloatTensor(raw[0]).to(self.device).reshape(self.seq_len, -1)
            if raw[1] not in self.support_labels and len(self.support_labels) < 5:
                self.support_labels.append(raw[1])
            y = torch.IntTensor([int(self.support_labels.index(raw[1]))]).to(self.device)
            self.support_set[y.item()] = x

    def export_onnx(self):
        input_names = ["support", "labels", "query"]
        output_names = ['pred']

        support = torch.randn((1, self.way*self.seq_len, self.n_joints*3), requires_grad=True)
        query = torch.randn((1, self.seq_len, self.n_joints*3), requires_grad=True)
        labels = torch.randn((1, self.way), requires_grad=True)
        onnx_file_name = "trx_{}_{}_{}_static.onnx".format(1, self.way*self.seq_len, self.n_joints*3)
        # Export the model
        print('Export the onnx model static ...')
        torch.onnx.export(self.model,
                          (support, labels, query),
                          onnx_file_name,
                          export_params=True,
                          opset_version=11,
                          do_constant_folding=True,
                          input_names=input_names, output_names=output_names,
                          dynamic_axes=None)

        print('Onnx model exporting done')
        return onnx_file_name


if __name__ == "__main__":
    from utils.params import TRXConfig
    from tqdm import tqdm
    from modules.hpe.hpe import HumanPoseEstimator
    from utils.params import MetrabsTRTConfig, RealSenseIntrinsics

    ar = ActionRecognizer(TRXConfig())
    ar.export_onnx()
    exit()

    h = HumanPoseEstimator(MetrabsTRTConfig(), RealSenseIntrinsics())

    ar.train((np.random.rand(16, 30, 3), "one"))
    ar.train((np.random.rand(16, 30, 3), "two"))
    ar.train((np.random.rand(16, 30, 3), "three"))
    ar.train((np.random.rand(16, 30, 3), "four"))
    ar.train((np.random.rand(16, 30, 3), "five"))

    p = np.random.rand(30, 3)

    for _ in tqdm(range(1000)):
        start = time.time()
        ar.inference(p)
        end = time.time()
        print(end - start)


# if __name__ == "__main__":
#
#     with open('assets/skeleton_types.pkl', "rb") as input_file:
#         skeleton_types = pickle.load(input_file)
#     skeleton = 'smpl+head_30'
#     edges = skeleton_types[skeleton]['edges']
#
#     ar = ActionRecognizer()
#     vis = PosePrinter(640, 480, just_pose=True)
#
#     with open("models/fake_data/shake_head.json", "rb") as infile:
#         query = json.load(infile)
#         query = np.array(query)[:, skeleton_types[skeleton]['indices'], :]/2200.  # Normalize
#         query -= query[:, 0][:, None, :]  # Center
#         query = torch.FloatTensor(query).cuda()
#         for elem in query:
#             vis.print_pose(elem.reshape(30, 3).detach().cpu().numpy(), edges)
#             time.sleep(0.2)
#         time.sleep(1)
#
#     with open("models/fake_data/pick2.json", "rb") as infile:
#         s1 = json.load(infile)
#         s1 = np.array(s1)[:, skeleton_types[skeleton]['indices'], :]/2200.
#         s1 -= s1[:, 0][:, None, :]  # Center
#         s1 = torch.FloatTensor(s1).cuda()
#         ar.train((s1, "pick"))
#
#     with open("models/fake_data/pointing.json", "rb") as infile:
#         s2 = json.load(infile)
#         s2 = np.array(s2)[:, skeleton_types[skeleton]['indices'], :]/2200.
#         s2 -= s2[:, 0][:, None, :]  # Center
#         s2 = torch.FloatTensor(s2).cuda()
#         ar.train((s2, "pointing"))
#
#     with open("models/fake_data/throw.json", "rb") as infile:
#         s3 = json.load(infile)
#         s3 = np.array(s3)[:, skeleton_types[skeleton]['indices'], :]/2200.
#         s3 -= s3[:, 0][:, None, :]  # Center
#         s3 = torch.FloatTensor(s3).cuda()
#         ar.train((s3, "throw"))
#
#     with open("models/fake_data/shake_head2.json", "rb") as infile:
#         s4 = json.load(infile)
#         s4 = np.array(s4)[:, skeleton_types[skeleton]['indices'], :]/2200.
#         s4 -= s4[:, 0][:, None, :]  # Center
#         s4 = torch.FloatTensor(s4).cuda()
#         ar.train((s4, "shake_head"))
#
#     with open("models/fake_data/hand_waving.json", "rb") as infile:
#         s5 = json.load(infile)
#         s5 = np.array(s5)[:, skeleton_types[skeleton]['indices'], :]/2200.
#         s5 -= s5[:, 0][:, None, :]  # Center
#         s5 = torch.FloatTensor(s5).cuda()
#         ar.train((s5, "hand_waving"))
#
#     for elem in query:
#         print(ar.inference(elem))
