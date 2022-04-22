import cv2
import torch
from modules.focus.mutual_gaze.head_detection.utils.misc import get_model
from modules.hpe.utils.misc import nms_cpu
from tqdm import tqdm


class HeadDetector:
    def __init__(self):
        self.model = get_model()
        self.model.load_state_dict(torch.load('modules/focus/mutual_gaze/head_detection/epoch_0.pth'))
        self.model.cuda()
        self.model.eval()

    def estimate(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        inp = torch.FloatTensor(img).cuda() / 255.
        inp = inp.permute(2, 0, 1)
        res = self.model([inp])
        boxes = res[0]['boxes'].detach().int().cpu().numpy()
        scores = res[0]['scores'].detach().cpu().numpy()
        good = nms_cpu(boxes, scores, nms_thresh=0.01)

        if len(good) > 0:
            boxes = boxes[good]
            scores = scores[good]
            good = scores > 0.8
            boxes = boxes[good]
            scores = scores[good]

        return boxes, scores


if __name__ == "__main__":
    cam = cv2.VideoCapture(0)
    detector = HeadDetector()

    for _ in tqdm(range(10000)):
        ret, img = cam.read()

        b, s = detector.estimate(img)

        if len(b) > 0:
            for box in b:
                img = cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

        cv2.imshow("", img)
        cv2.waitKey(1)
