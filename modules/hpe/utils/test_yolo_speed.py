import copy
import cv2
import pickle
from modules.hpe.utils.misc import postprocess_yolo_output, homography, get_augmentations, is_within_fov, reconstruct_absolute
import einops
import numpy as np
from utils.input import RealSense
from utils.tensorrt_runner import Runner


class HumanPoseEstimator:
    def __init__(self, model_config, cam_config):

        self.yolo_thresh = model_config.yolo_thresh
        self.nms_thresh = model_config.nms_thresh
        self.num_aug = model_config.num_aug

        # Load modules
        self.yolo = Runner(model_config.yolo_engine_path)

    def estimate(self, frame):
        # Preprocess for yolo
        square_img = cv2.resize(frame, (256, 256), fx=1.0, fy=1.0, interpolation=cv2.INTER_AREA)
        yolo_in = copy.deepcopy(square_img)
        yolo_in = cv2.cvtColor(yolo_in, cv2.COLOR_BGR2RGB)
        yolo_in = np.transpose(yolo_in, (2, 0, 1)).astype(np.float32)
        yolo_in = np.expand_dims(yolo_in, axis=0)
        yolo_in = yolo_in / 255.0

        # Yolo
        outputs = self.yolo(yolo_in)
        boxes, confidences = outputs[0].reshape(1, 4032, 1, 4), outputs[1].reshape(1, 4032, 80)
        bboxes_batch = postprocess_yolo_output(boxes, confidences, self.yolo_thresh, self.nms_thresh)
        return


if __name__ == "__main__":
    from utils.params import MetrabsTRTConfig, RealSenseIntrinsics, MainConfig
    from tqdm import tqdm
    import cv2

    cap = cv2.VideoCapture('assets/dance.mp4')

    def test(pytorch):
        if pytorch:
            import sys
            sys.path.append("modules/hpe/assets/pytorchYOLOv4")
            from modules.hpe.assets.pytorchYOLOv4.models import Yolov4
            import torch
            N_CLASSES = 80
            model = Yolov4(n_classes=N_CLASSES, inference=True)
            WEIGHT_FILE = "modules/hpe/modules/raws/yolov4.pth"


            def rewrite(mod, weight_path, output_path):
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                weight = torch.load(weight_path, map_location=device)

                weight_dict = {}
                for key, val in weight.items():
                    if 'neek' in key:
                        key = key.replace("neek", "neck")
                    weight_dict.update({key: val})
                mod.load_state_dict(weight_dict)
                torch.save(mod.state_dict(), output_path)

            rewrite(model, WEIGHT_FILE, WEIGHT_FILE)

            pretrained_dict = torch.load(WEIGHT_FILE, map_location=torch.device('cuda'))
            model.load_state_dict(pretrained_dict)
            model.cuda()
            model.eval()
            for _ in tqdm(range(100)):
                ret, img = cap.read()
                if not ret:
                    break
                img = cv2.resize(img, (640, 480))
                img = torch.FloatTensor(img).to('cuda').permute(2, 1, 0).unsqueeze(0)
                model(img)
        else:
            args = MainConfig()

            h = HumanPoseEstimator(MetrabsTRTConfig(), RealSenseIntrinsics())

            for _ in tqdm(range(100)):
                ret, img = cap.read()
                if not ret:
                    break
                img = cv2.resize(img, (640, 480))
                h.estimate(img)

    test(False)
