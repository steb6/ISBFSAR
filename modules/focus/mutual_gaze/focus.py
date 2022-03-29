from modules.focus.mutual_gaze.head_detection.utils.misc import get_model
from modules.hpe.utils.misc import nms_cpu
import torch.optim
import copy
from torchvision import transforms
from modules.focus.mutual_gaze.focus_detection.model import MutualGazeDetector
from tqdm import tqdm
import cv2
import numpy as np


if __name__ == "__main__":
    head_model = get_model()
    head_model.load_state_dict(torch.load('modules/focus/mutual_gaze/head_detection/epoch_0.pth'))
    head_model.cuda()
    head_model.eval()

    focus_model = MutualGazeDetector()
    focus_model.load_state_dict(torch.load('modules/focus/mutual_gaze/focus_detection/0.98.pth'))
    focus_model.cuda()
    focus_model.eval()

    cam = cv2.VideoCapture(2)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # for _ in tqdm(range(10000)):
    while True:
        ret, img = cam.read()

        inp = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        inp = torch.FloatTensor(inp).cuda() / 255.
        inp = inp.permute(2, 0, 1)
        res = head_model([inp])
        boxes = res[0]['boxes'].detach().int().cpu().numpy()
        scores = res[0]['scores'].detach().cpu().numpy()
        good = nms_cpu(boxes, scores, nms_thresh=0.01)

        if len(good) > 0:
            boxes = boxes[good]
            scores = scores[good]
            good = scores > 0.8
            boxes = boxes[good]
            scores = scores[good]

        if len(boxes) > 0:
            box = boxes[0]
            x = img[box[0]:box[2], box[1]:box[3]]

            # TODO PAD
            if x.shape[0] < x.shape[1]:
                pad = int((x.shape[1] - x.shape[0]) / 2)
                x = np.pad(x, ((pad, pad), (0, 0), (0, 0)), 'constant', constant_values=0)
            elif x.shape[1] < x.shape[0]:
                pad = int((x.shape[0] - x.shape[1]) / 2)
                x = np.pad(x, ((0, 0), (pad, pad), (0, 0)), 'constant', constant_values=0)
            x = cv2.resize(x, (256, 256))
            # TODO END PAD

            x = torch.FloatTensor(x)
            x = x.unsqueeze(0)
            x = x.permute(0, 3, 1, 2)
            x = x / 255.
            x = normalize(x)
            x = x.cuda()

            out = focus_model(x)

            img = cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            print("FOCUS" if out.item() > 0.5 else "NOT FOCUS")
            cv2.putText(img, "{}".format(out.item()), (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0),
                        2,
                        cv2.LINE_AA)

        cv2.imshow("bbox", img)
        cv2.waitKey(1)
