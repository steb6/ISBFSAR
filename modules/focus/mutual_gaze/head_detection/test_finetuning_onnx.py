import cv2
import torch
from modules.hpe.utils.misc import nms_cpu
from onnxruntime import InferenceSession
from tqdm import tqdm
import numpy as np

if __name__ == "__main__":
    model = InferenceSession('modules/focus/head_detection/modules/onnx/longest.onnx')
    cam = cv2.VideoCapture(0)

    for _ in tqdm(range(10000)):
        ret, img = cam.read()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        inp = np.transpose(img, (2, 0, 1)).astype(np.float32) / 255.
        res = model.run(None, {'img': inp[None, ...]})
        boxes = res[0].astype(int)
        scores = res[2]
        good = nms_cpu(boxes, scores, nms_thresh=0.01)

        if len(good) > 0:
            boxes = boxes[good]
            scores = scores[good]
            good = scores > 0.8
            boxes = boxes[good]
            scores = scores[good]
            if len(boxes) > 0:
                for box in boxes:
                    img = cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

        cv2.imshow("", img)
        cv2.waitKey(1)
