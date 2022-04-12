import cv2
import numpy as np
import torch
from modules.focus.mutual_gaze.focus_detection.openpose_utilities import read_openpose_from_json, get_features
from modules.focus.mutual_gaze.focus_detection.utilities import get_eye_bbox_openpose
from modules.focus.mutual_gaze.head_detection.utils.misc import get_model
from tqdm import tqdm
import shutil
import os
import json


if __name__ == "__main__":
    base_path = 'D:/datasets/mutualGaze_dataset/realsense'
    out_path = 'D:/datasets/focus_dataset'

    normal = base_path + '/eyecontact_annotations.txt'
    moving = base_path + '/eyecontact_annotations_moving_head.txt'
    rotating = base_path + '/eyecontact_annotations_rotate_head_body.txt'

    model = get_model()
    model.load_state_dict(torch.load('modules/focus/mutual_gaze/head_detection/epoch_0.pth'))
    model.cuda()
    model.eval()

    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    os.mkdir(out_path)

    for elem in [normal, moving, rotating]:
        with open(elem, "r") as in_file:
            lines = in_file.readlines()

        lines = [line.split(' ') for line in lines]
        lines = [(base_path + line[0][1:], bool(int(line[1][0]))) for line in lines]

        i = 0
        for line in tqdm(lines):
            img = cv2.imread(line[0])
            json_file = line[0].replace('eyecontact_images_human',
                                        'eyecontact_data_openpose').replace('.jpg', '_keypoints.json')
            openpose = read_openpose_from_json(json_file)
            joint_pose, joint_confs, keypoints, confidences = openpose
            assert len(keypoints) == 1
            features = get_features(joint_pose, joint_confs, keypoints, confidences)
            eye_bbox = get_eye_bbox_openpose(joint_pose[0], joint_confs[0])

            # TODO DEBUG
            # keypoints = keypoints[0]
            # for point in joint_pose[0]:
            #     img = cv2.circle(img, (int(point[0]), int(point[1])), 1, (0, 0, 255), 1, cv2.LINE_AA)
            # for point in keypoints:
            #     img = cv2.circle(img, (int(point[0]), int(point[1])), 1, (255, 0, 0), 1, cv2.LINE_AA)
            # img = cv2.rectangle(img, (eye_bbox[0], eye_bbox[1]), (eye_bbox[2], eye_bbox[3]), (0, 255, 0), 1, cv2.LINE_AA)
            # cv2.imshow("", img)
            # cv2.waitKey(0)
            # TODO END DEBUG

            inp = img[eye_bbox[1]:eye_bbox[3], eye_bbox[0]:eye_bbox[2]]
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
            offset = int(len(os.listdir(out_dir)) / 2)

            cv2.imwrite(out_dir + '/{}.jpg'.format(offset), x)

            with open(out_dir + '/{}.json'.format(offset), 'w') as outfile:

                keypoints = [(float(elem[0]), float(elem[1])) for elem in keypoints[0]]
                confidences = list(map(float, confidences[0]))

                json.dump({"features": features[0][2:].tolist(),
                           "keypoints": keypoints[0],
                           "confidences": confidences[0]}, outfile)

            i += 1
