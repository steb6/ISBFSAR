from modules.hpe.utils.misc import homography
import numpy as np
import torch
import onnx
import onnxruntime as ort


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

        # Load modules
        onnx_model = onnx.load("modules/hpe/modules/onnxs/image_transformation1.onnx")
        onnx.checker.check_model(onnx_model)

    def estimate(self, frame):
        # Preprocess for BackBone
        x1 = 100
        y1 = 100
        x2 = 200
        y2 = 200
        new_K, homo_inv = homography(x1, x2, y1, y2, self.K, 256)

        # Apply homography
        H = self.K @ np.linalg.inv(new_K @ homo_inv)
        ort_sess = ort.InferenceSession("modules/hpe/modules/onnxs/image_transformation1.onnx")
        outputs = ort_sess.run(None, {'frame': frame.astype(np.int32),
                                      "H": H.astype(np.float32)})[0]

        cv2.imshow("img", frame)
        cv2.imshow("second", outputs[0].astype(np.uint8))
        cv2.waitKey(1)


if __name__ == "__main__":
    from utils.params import MetrabsTRTConfig, RealSenseIntrinsics, MainConfig
    from tqdm import tqdm
    import cv2

    args = MainConfig()
    h = HumanPoseEstimator(MetrabsTRTConfig(), RealSenseIntrinsics())
    img = cv2.imread("assets/input.jpg")

    for _ in tqdm(range(10000)):
        h.estimate(img)
