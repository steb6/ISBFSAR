import sys, os, argparse

import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision
import torch.nn.functional as F
from PIL import Image

from deepheadpose.code import datasets, hopenet, utils

from skimage import io
import dlib

from modules.focus.mutual_gaze.head_detector import HeadDetector


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Head pose estimation using the Hopenet network.')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--snapshot', dest='snapshot', help='Path of model snapshot.',
                        default='', type=str)
    parser.add_argument('--face_model', dest='face_model', help='Path of DLIB face detection model.',
                        default='', type=str)
    parser.add_argument('--video', dest='video_path', help='Path of video')
    parser.add_argument('--output_string', dest='output_string', help='String appended to output file')
    parser.add_argument('--n_frames', dest='n_frames', help='Number of frames', type=int)
    parser.add_argument('--fps', dest='fps', help='Frames per second of source video', type=float, default=30.)
    args = parser.parse_args()
    return args


class HeadPoseDetector:
    def __init__(self):
        cudnn.enabled = True

        batch_size = 1
        self.gpu = 0
        snapshot_path = 'modules/focus/gaze_estimation/modules/raw/hopenet_robust_alpha1.pkl'

        # ResNet50 structure
        self.model = hopenet.Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)

        # Dlib face detection model
        self.face_detector = HeadDetector()

        print('Loading snapshot.')
        # Load snapshot
        saved_state_dict = torch.load(snapshot_path)
        self.model.load_state_dict(saved_state_dict)

        print('Loading data.')

        self.transformations = transforms.Compose([transforms.Scale(224),
                                              transforms.CenterCrop(224), transforms.ToTensor(),
                                              transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                   std=[0.229, 0.224, 0.225])])

        self.model.cuda(self.gpu)

        print('Ready to test network.')

        # Test the Model
        self.model.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
        total = 0

        idx_tensor = [idx for idx in range(66)]
        self.idx_tensor = torch.FloatTensor(idx_tensor).cuda(self.gpu)

        # New cv2
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))  # float
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float

        frame_num = 1

    def estimate(self, frame):

        bboxes, scores = self.face_detector.estimate(frame)

        cv2_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        for box, conf in zip(bboxes, scores):
            x_min, y_min, x_max, y_max = box

            if conf > 0.8:
                bbox_width = abs(x_max - x_min)
                bbox_height = abs(y_max - y_min)
                x_min -= 2 * bbox_width / 4
                x_max += 2 * bbox_width / 4
                y_min -= 3 * bbox_height / 4
                y_max += bbox_height / 4
                x_min = max(x_min, 0)
                y_min = max(y_min, 0)
                x_max = min(frame.shape[1], x_max)
                y_max = min(frame.shape[0], y_max)
                # Crop image
                img = cv2_frame[int(y_min):int(y_max), int(x_min):int(x_max)]
                img = Image.fromarray(img)

                # Transform
                img = self.transformations(img)
                img_shape = img.size()
                img = img.view(1, img_shape[0], img_shape[1], img_shape[2])
                img = Variable(img).cuda(self.gpu)

                yaw, pitch, roll = self.model(img)

                yaw_predicted = F.softmax(yaw)
                pitch_predicted = F.softmax(pitch)
                roll_predicted = F.softmax(roll)
                # Get continuous predictions in degrees.
                yaw_predicted = torch.sum(yaw_predicted.data[0] * self.idx_tensor) * 3 - 99
                pitch_predicted = torch.sum(pitch_predicted.data[0] * self.idx_tensor) * 3 - 99
                roll_predicted = torch.sum(roll_predicted.data[0] * self.idx_tensor) * 3 - 99

                # Print new frame with cube and axis
                # utils.plot_pose_cube(frame, yaw_predicted, pitch_predicted, roll_predicted, (x_min + x_max) / 2, (y_min + y_max) / 2, size = bbox_width)

                # Plot expanded bounding box
                # cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0,255,0), 1)

                return yaw_predicted, roll_predicted, pitch_predicted, x_min, y_min, x_max, y_max


if __name__ == '__main__':

    video = cv2.VideoCapture(0)
    detector = HeadPoseDetector()

    for frame_num in tqdm(range(1000)):

        ret, frame = video.read()
        if not ret:
            break

        res = detector.estimate(frame)

        if res is not None:
            yaw, pitch, roll, x_min, y_min, x_max, y_max = res
            frame = utils.draw_axis(frame, yaw, pitch, roll, tdx=(x_min + x_max) / 2, tdy=(y_min + y_max) / 2)
            frame = cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)

        cv2.imshow("", frame)
        cv2.waitKey(1)

    video.release()
