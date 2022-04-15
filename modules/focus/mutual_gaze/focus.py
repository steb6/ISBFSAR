from modules.focus.mutual_gaze.head_detection.utils.misc import get_model
from modules.hpe.utils.misc import nms_cpu
import torch.optim
import copy
from torchvision import transforms
from modules.focus.mutual_gaze.focus_detection.utils.model import MutualGazeDetectorHeads as Model
import cv2
import numpy as np
from tqdm import tqdm

WINDOW_SIZE = 3

if __name__ == "__main__":
    head_model = get_model()
    head_model.load_state_dict(torch.load('modules/focus/mutual_gaze/head_detection/epoch_0.pth'))
    head_model.cuda()
    head_model.eval()

    focus_model = Model()
    focus_model.load_state_dict(torch.load('modules/focus/mutual_gaze/focus_detection/checkpoints/MNET3/sess_1_acc_0.80.pth'))
    focus_model.cuda()
    focus_model.eval()

    # cam = cv2.VideoCapture('assets/test_gaze_with_mask.mp4')
    cam = cv2.VideoCapture('video.mp4')
    # cam = RealSense(1920, 1080)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    focuses = []
    for _ in tqdm(range(10000)):
    # while True:

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
            x = img[box[1]:box[3], box[0]:box[2]]

            # TODO PAD
            if x.shape[0] < x.shape[1]:
                pad = int((x.shape[1] - x.shape[0]) / 2)
                x = np.pad(x, ((pad, pad), (0, 0), (0, 0)), 'constant', constant_values=0)
            elif x.shape[1] < x.shape[0]:
                pad = int((x.shape[0] - x.shape[1]) / 2)
                x = np.pad(x, ((0, 0), (pad, pad), (0, 0)), 'constant', constant_values=0)
            x = cv2.resize(x, (256, 256))
            # TODO END PAD

            normalized_image = copy.deepcopy(x)

            # TODO TEST TIME AUGMENTATION
            # imgs = [x for _ in range(8)]
            # imgs_aug = []
            # for i in imgs:
            #     i = horizontal_shift(i, 0.2)
            #     i = vertical_shift(i, 0.2)
            #     i = brightness(i, 0.5, 2)
            #     i = zoom(i, 0.9)
            #     if random.random() < 0.5:
            #         i = horizontal_flip(i, False)
            #     i = rotation(i, 30)
            #     imgs_aug.append(i)
            # x = np.stack(imgs_aug)
            # TODO TEST TIME AUGMENTATION

            x = torch.FloatTensor(x)
            x = x.unsqueeze(0)
            x = x.permute(0, 3, 1, 2)
            x = x / 255.
            x = normalize(x)
            x = x.cuda()

            out = focus_model(x)
            out = out.mean()
            focuses.append(out.item() > 0.5)
            if WINDOW_SIZE > 1:
                if len(focuses) > WINDOW_SIZE:
                    focuses = focuses[-WINDOW_SIZE:]
                is_focus = sum(focuses) > (len(focuses) / 2)
            else:
                is_focus = out.item() > 0.5

            color = (0, 0, 255) if is_focus < 0.5 else (0, 255, 0)
            img = cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color, 2)
            cv2.putText(img, "{:.2f}".format(out.item()), (box[0], box[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0),
                        2,
                        cv2.LINE_AA)

            cv2.imshow("normalized", normalized_image)
            cv2.imshow("bbox", img)
            cv2.waitKey(1)
