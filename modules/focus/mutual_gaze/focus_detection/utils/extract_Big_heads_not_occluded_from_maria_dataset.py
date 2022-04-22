import cv2
import numpy as np
import torch
from modules.focus.mutual_gaze.head_detection.utils.misc import get_model
from modules.hpe.utils.misc import nms_cpu
from tqdm import tqdm
import shutil
import os


if __name__ == "__main__":
    base_path = 'D:/datasets/mutualGaze_dataset/realsense'
    out_path = 'D:/datasets/focus_dataset_BIG_heads_no_occluded'

    normal = base_path + '/eyecontact_annotations.txt'
    # moving = base_path + '/eyecontact_annotations_moving_head.txt'
    rotating = base_path + '/eyecontact_annotations_rotate_head_body.txt'

    model = get_model()
    model.load_state_dict(torch.load('modules/focus/mutual_gaze/head_detection/epoch_0.pth'))
    model.cuda()
    model.eval()

    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    os.mkdir(out_path)

    for elem in [normal, rotating]:
        with open(elem, "r") as in_file:
            lines = in_file.readlines()

        lines = [line.split(' ') for line in lines]
        lines = [(base_path + line[0][1:], bool(int(line[1][0]))) for line in lines]

        for line in tqdm(lines):
            img = cv2.imread(line[0])
            inp = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            inp = torch.FloatTensor(inp).cuda() / 255.
            inp = inp.permute(2, 0, 1)
            res = model([inp])
            boxes = res[0]['boxes'].detach().int().cpu().numpy()
            scores = res[0]['scores'].detach().cpu().numpy()
            good = nms_cpu(boxes, scores, nms_thresh=0.01)

            if len(good) > 0:
                boxes = boxes[good]
                scores = scores[good]
                good = scores > 0.9
                boxes = boxes[good]
                scores = scores[good]
                if len(boxes) > 0:
                    for box in boxes:
                        # TODO START BIG HEADS
                        x_min, y_min, x_max, y_max = box
                        delta_y = int((y_max - y_min) / 4)
                        y_max = min(480, y_max + delta_y)
                        # cv2.imshow("",  inp)  # TODO DEBUG
                        # cv2.waitKey(0)  # TODO DEBUG
                        # TODO END
                        inp = img[y_min:y_max, x_min:x_max]
                        if inp.shape[0] < inp.shape[1]:
                            pad = int((inp.shape[1] - inp.shape[0]) / 2)
                            inp = np.pad(inp, ((pad, pad), (0, 0), (0, 0)), 'constant', constant_values=0)
                        elif inp.shape[1] < inp.shape[0]:
                            pad = int((inp.shape[0] - inp.shape[1]) / 2)
                            inp = np.pad(inp, ((0, 0), (pad, pad), (0, 0)), 'constant', constant_values=0)
                        inp = cv2.resize(inp, (256, 256))

                        # Save image in right place
                        x, y = inp, line[1]
                        session = line[0].split('/')[-3]
                        out_dir = out_path + '/{}'.format(session)
                        if not os.path.exists(out_dir):
                            os.mkdir(out_dir)
                        out_dir = out_dir + ('/watching' if y else '/not_watching')
                        if not os.path.exists(out_dir):
                            os.mkdir(out_dir)
                        offset = len(os.listdir(out_dir))

                        cv2.imwrite(out_dir + '/{}.jpg'.format(offset), x)
