from modules.focus.mutual_gaze.head_detection.utils.misc import get_model
from modules.hpe.utils.misc import nms_cpu
import torch.optim
import copy
from torchvision import transforms
from modules.focus.mutual_gaze.focus_detection.utils.model import MutualGazeDetectorHeads as Model
import cv2
import numpy as np
from tqdm import tqdm
from utils.params import MutualGazeConfig

WINDOW_SIZE = 1


class FocusDetector:
    def __init__(self, args):
        self.head_model = get_model()
        self.head_model.load_state_dict(torch.load(args.head_model))
        self.head_model.cuda()
        self.head_model.eval()

        self.focus_model = Model(args.model)
        self.focus_model.load_state_dict(
            torch.load(args.focus_model))
        self.focus_model.cuda()
        self.focus_model.eval()

        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.focuses = []

    def estimate(self, img):

        # Get head candidates
        inp = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        inp = torch.FloatTensor(inp).cuda() / 255.
        inp = inp.permute(2, 0, 1)
        res = self.head_model([inp])
        boxes = res[0]['boxes'].detach().int().cpu().numpy()
        scores = res[0]['scores'].detach().cpu().numpy()
        good = nms_cpu(boxes, scores, nms_thresh=0.01)

        # Filter candidates
        if len(good) > 0:
            boxes = boxes[good]
            scores = scores[good]
            good = scores > 0.8
            boxes = boxes[good]
            scores = scores[good]

        # Take best candidate and postprocess image
        if len(boxes) > 0:
            box = boxes[0]
            x = img[box[1]:box[3], box[0]:box[2]]

            if x.shape[0] < x.shape[1]:
                pad = int((x.shape[1] - x.shape[0]) / 2)
                x = np.pad(x, ((pad, pad), (0, 0), (0, 0)), 'constant', constant_values=0)
            elif x.shape[1] < x.shape[0]:
                pad = int((x.shape[0] - x.shape[1]) / 2)
                x = np.pad(x, ((0, 0), (pad, pad), (0, 0)), 'constant', constant_values=0)

            # Preprocessing for focus
            x = cv2.resize(x, (224, 224))
            x = x / 255.
            x = x - np.array([0.485, 0.456, 0.406])
            x = x / np.array([0.229, 0.224, 0.225])
            x = torch.FloatTensor(x).cuda()
            x = x.unsqueeze(0)
            x = x.permute(0, 3, 1, 2)

            out = self.focus_model(x)
            out = out.mean()
            self.focuses.append(out.item() > 0.5)
            if WINDOW_SIZE > 1:
                if len(self.focuses) > WINDOW_SIZE:
                    self.focuses = self.focuses[-WINDOW_SIZE:]
                is_focus = sum(self.focuses) > (len(self.focuses) / 2)
            else:
                is_focus = out.item() > 0.5

            return out.item(), box, is_focus

    def print_bbox(self, img, box, score, is_focus):
        img = cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0) if is_focus else (0, 0, 255), 2)
        img = cv2.putText(img, "Focus: {:.2f}".format(score), (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 1,
                          (0, 255, 0) if is_focus else (0, 0, 255))
        return img


if __name__ == "__main__":

    cam = cv2.VideoCapture('assets/recording.mp4')
    # cam = cv2.VideoCapture(0)

    focus_detector = FocusDetector(MutualGazeConfig())

    for _ in tqdm(range(10000)):
        ret, img = cam.read()
        res = focus_detector.estimate(img)
        if res:
            score, bbox, is_focus = res

            img = focus_detector.print_bbox(img, bbox, score, is_focus)
            print(score)
            cv2.imshow("bbox", img)
            cv2.waitKey(1)
