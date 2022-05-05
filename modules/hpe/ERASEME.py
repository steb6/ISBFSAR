import numpy as np
from utils.tensorrt_runner import Runner
import time


class HumanPoseEstimator:
    def __init__(self, model_config):
        self.bbone = Runner(model_config.bbone_engine_path)
        self.image_transformation = Runner(model_config.image_transformation_path)
        self.yolo = Runner(model_config.yolo_engine_path)

    def estimate(self):
        # BackBone
        start_bbone = time.time()
        _ = self.yolo(np.random.random((256, 256, 3)))
        print(time.time() - start_bbone)
        _ = self.bbone(np.random.random((1, 256, 256, 3)))
        _ = self.image_transformation([np.random.random((480, 640, 3)).astype(int),
                                       np.random.random((1, 3, 3))])


if __name__ == "__main__":
    from utils.params import MetrabsTRTConfig
    from tqdm import tqdm

    h = HumanPoseEstimator(MetrabsTRTConfig())

    for _ in tqdm(range(10000)):
        # while True:
        h.estimate()
