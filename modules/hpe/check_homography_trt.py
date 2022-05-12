import pickle
from modules.hpe.utils.misc import homography
import numpy as np
from utils.tensorrt_runner import Runner


class HumanPoseEstimator:
    def __init__(self, model_config, cam_config):

        self.yolo_thresh = model_config.yolo_thresh
        self.nms_thresh = model_config.nms_thresh
        self.num_aug = model_config.num_aug
        self.n_test = 1 if self.num_aug < 1 else self.num_aug

        # Intrinsics and K matrix of RealSense
        self.K = np.zeros((3, 3), np.float32)
        self.K[0][0] = cam_config.fx
        self.K[0][2] = cam_config.ppx
        self.K[1][1] = cam_config.fy
        self.K[1][2] = cam_config.ppy
        self.K[2][2] = 1

        # Load conversions
        self.skeleton = model_config.skeleton
        self.expand_joints = np.load(model_config.expand_joints_path)
        with open(model_config.skeleton_types_path, "rb") as input_file:
            self.skeleton_types = pickle.load(input_file)

        # Load modules
        self.image_transformation = Runner(model_config.image_transformation_path)

    def estimate(self, frame):

        # Preprocess for BackBone
        x1 = 100
        y1 = 100
        x2 = 200
        y2 = 200
        new_K, homo_inv = homography(x1, x2, y1, y2, self.K, 256)

        # Apply homography
        H = self.K @ np.linalg.inv(new_K @ homo_inv)
        bbone_in = self.image_transformation([frame.astype(np.int32), H.astype(np.float32)])

        bbone_in = bbone_in[0].reshape(self.n_test, 256, 256, 3)
        cv2.imshow("BBONE", bbone_in[0].astype(np.uint8))  # TODO SOLVE THE MISTERY
        cv2.imshow("img", frame)
        cv2.waitKey(1)  # TODO SOLVE THE MISTERY


if __name__ == "__main__":
    from utils.params import MetrabsTRTConfig, RealSenseIntrinsics, MainConfig
    from tqdm import tqdm
    import cv2

    args = MainConfig()
    h = HumanPoseEstimator(MetrabsTRTConfig(), RealSenseIntrinsics())
    img = cv2.imread("assets/input.jpg")

    for _ in tqdm(range(10000)):
        h.estimate(img)

