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

        # Intrinsics and K matrix of RealSense
        self.K = np.zeros((3, 3), np.float32)
        self.K[0][0] = cam_config.fx
        self.K[0][2] = cam_config.ppx
        self.K[1][1] = cam_config.fy
        self.K[1][2] = cam_config.ppy
        self.K[2][2] = 1

        # SMPL+HEAD_30 mirror mapping
        self.joint_mirror_map = np.array([[1, 2],
                                          [4, 5],
                                          [7, 8],
                                          [10, 11],
                                          [13, 14],
                                          [16, 17],
                                          [18, 19],
                                          [20, 21],
                                          [22, 23],
                                          [25, 26],
                                          [28, 29]])

        # Load conversions
        self.skeleton = model_config.skeleton
        self.expand_joints = np.load(model_config.expand_joints_path)
        with open(model_config.skeleton_types_path, "rb") as input_file:
            self.skeleton_types = pickle.load(input_file)

        # Load modules
        self.yolo = Runner(model_config.yolo_engine_path)
        # with open(model_config.yolo_engine_path, 'rb') as file:
        #     self.yolo_engine = EngineFromBytes(file.read())
        # self.yolo = TrtRunner(self.yolo_engine)
        # self.yolo.activate()

        self.image_transformation = Runner(model_config.image_transformation_path)
        # with open(model_config.image_transformation_path, 'rb') as file:
        #     self.image_transformation_engine = EngineFromBytes(file.read())
        # self.image_transformation = TrtRunner(self.image_transformation_engine)
        # self.image_transformation.activate()

        self.bbone = Runner(model_config.bbone_engine_path)
        # with open(model_config.bbone_engine_path, 'rb') as file:
        #     self.bbone_engine = EngineFromBytes(file.read())
        # self.bbone = TrtRunner(self.bbone_engine)
        # self.bbone.activate()

        self.head_weights = np.load(model_config.head_weight_path)
        self.heads_bias = np.load(model_config.head_bias_path)

    def estimate(self, frame):
        # Preprocess for yolo
        square_img = cv2.resize(frame, (256, 256), fx=1.0, fy=1.0, interpolation=cv2.INTER_AREA)
        yolo_in = copy.deepcopy(square_img)
        yolo_in = cv2.cvtColor(yolo_in, cv2.COLOR_BGR2RGB)
        yolo_in = np.transpose(yolo_in, (2, 0, 1)).astype(np.float32)
        yolo_in = np.expand_dims(yolo_in, axis=0)
        yolo_in = yolo_in / 255.0

        # Yolo
        # outputs = self.yolo.infer(feed_dict={"input": yolo_in})
        # boxes, confidences = outputs['boxes'], outputs['confs']
        outputs = self.yolo(yolo_in)
        boxes, confidences = outputs[0].reshape(1, 4032, 1, 4), outputs[1].reshape(1, 4032, 80)
        bboxes_batch = postprocess_yolo_output(boxes, confidences, self.yolo_thresh, self.nms_thresh)

        # Get only the bounding box with the human with highest probability
        box = bboxes_batch[0]  # Remove batch dimension
        humans = []
        for e in box:  # For each object in the image
            if e[5] == 0:  # If it is a human
                humans.append(e)
        if len(humans) > 0:
            humans.sort(key=lambda x: x[4], reverse=True)  # Sort with decreasing probability
            human = humans[0]
        else:
            return None, None, None, None, None, None, None, None

        # Preprocess for BackBone
        x1 = int(human[0] * frame.shape[1]) if int(human[0] * frame.shape[1]) > 0 else 0
        y1 = int(human[1] * frame.shape[0]) if int(human[1] * frame.shape[0]) > 0 else 0
        x2 = int(human[2] * frame.shape[1]) if int(human[2] * frame.shape[1]) > 0 else 0
        y2 = int(human[3] * frame.shape[0]) if int(human[3] * frame.shape[0]) > 0 else 0
        new_K, homo_inv = homography(x1, x2, y1, y2, self.K, 256)

        # Test time augmentation TODO ADD GAMMA DECODING
        aug_should_flip, aug_rotflipmat, aug_gammas, aug_scales = get_augmentations(self.num_aug)
        new_K = np.tile(new_K, (self.num_aug, 1, 1))
        for k in range(self.num_aug):
            new_K[k, :2, :2] *= aug_scales[k]
        homo_inv = aug_rotflipmat @ np.tile(homo_inv[0], (self.num_aug, 1, 1))

        # Apply homography
        H = self.K @ np.linalg.inv(new_K @ homo_inv)
        # bbone_in = self.image_transformation.infer(feed_dict={"frame": frame.astype(int),
        #                                                       "H": H.astype(np.float32)})
        # bbone_in = bbone_in['images']
        bbone_in = self.image_transformation([frame.astype(int), H.astype(np.float32)])
        bbone_in = bbone_in[0].reshape(5, 256, 256, 3)
        bbone_in_ = (bbone_in / 255.0).astype(np.float32)

        # BackBone
        # outputs = self.bbone.infer(feed_dict={"images": bbone_in_})
        # features = outputs['prediction']
        outputs = self.bbone(bbone_in_)
        features = outputs[0].reshape(5, 8, 8, 1280)

        # Heads
        logits = (features @ self.head_weights[0][0]) + self.heads_bias  # 5, 8, 8, 288

        # Get logits 3d  TODO DO THE SAME WITH 2D
        _, logits2d, logits3d = np.split(logits, [0, 32], axis=3)
        current_format = 'b h w (d j)'
        logits3d = einops.rearrange(logits3d, f'{current_format} -> b h w d j', j=32)  # 5, 8, 8, 9, 32

        # 3D Softmax
        heatmap_axes = (2, 1, 3)
        max_along_axis = logits3d.max(axis=heatmap_axes, keepdims=True)
        exponential = np.exp(logits3d - max_along_axis)
        denominator = np.sum(exponential, axis=heatmap_axes, keepdims=True)
        res = exponential / denominator

        # 3D Decode Heatmap
        result = []
        for ax in heatmap_axes:
            other_heatmap_axes = tuple(other_ax for other_ax in heatmap_axes if other_ax != ax)
            summed_over_other_heatmap_axes = np.sum(res, axis=other_heatmap_axes, keepdims=True)
            coords = np.linspace(0.0, 1.0, res.shape[ax])
            decoded = np.tensordot(summed_over_other_heatmap_axes, coords, axes=[[ax], [0]])
            result.append(np.squeeze(np.expand_dims(decoded, ax), axis=heatmap_axes))
        pred3d = np.stack(result, axis=-1)

        # 2D Softmax
        heatmap_axes = (2, 1)
        max_along_axis = logits2d.max(axis=heatmap_axes, keepdims=True)
        exponential = np.exp(logits2d - max_along_axis)
        denominator = np.sum(exponential, axis=heatmap_axes, keepdims=True)
        res = exponential / denominator

        # Decode heatmap
        result = []
        for ax in heatmap_axes:
            other_heatmap_axes = tuple(other_ax for other_ax in heatmap_axes if other_ax != ax)
            summed_over_other_heatmap_axes = np.sum(res, axis=other_heatmap_axes, keepdims=True)
            coords = np.linspace(0.0, 1.0, res.shape[ax])
            decoded = np.tensordot(summed_over_other_heatmap_axes, coords, axes=[[ax], [0]])
            result.append(np.squeeze(np.expand_dims(decoded, ax), axis=heatmap_axes))
        pred2d = np.stack(result, axis=-1) * 255

        # Get absolute position (if desired)
        is_predicted_to_be_in_fov = is_within_fov(pred2d)

        # If less than 1/4 of the joints is visible, then the resulting pose will be weird
        if is_predicted_to_be_in_fov.sum() < is_predicted_to_be_in_fov.size/4:
            return None, None, None, None, None, None, None, None

        # Move the skeleton into estimated absolute position if necessary  # TODO ADD AGAIN
        # pred3d = reconstruct_absolute(pred2d, pred3d, new_K, is_predicted_to_be_in_fov, weak_perspective=False)

        # Go back in original space (without augmentation and homography)
        pred3d = pred3d @ homo_inv

        # TODO TRY 1
        # pred2d_original = np.concatenate((pred2d, np.ones_like(pred2d)[..., :1]), axis=-1) @ homo_inv
        # pred2d_original = pred2d_original[..., :2]
        # TODO TRY 2
        pred2d_original = np.concatenate((pred2d, np.ones_like(pred2d)[..., :1]), axis=-1) @ homo_inv
        pred2d_original = pred2d_original[..., :2]
        # TODO END TRY

        # Get correct skeleton
        pred3d = (pred3d.swapaxes(1, 2) @ self.expand_joints).swapaxes(1, 2)
        if self.skeleton is not None:
            pred3d = pred3d[:, self.skeleton_types[self.skeleton]['indices']]
            edges = self.skeleton_types[self.skeleton]['edges']
        else:
            edges = None

        # Mirror predictions
        for k in range(len(self.joint_mirror_map)):
            aux = pred3d[aug_should_flip, self.joint_mirror_map[k, 0]]
            pred3d[aug_should_flip, self.joint_mirror_map[k, 0]] = pred3d[aug_should_flip, self.joint_mirror_map[k, 1]]
            pred3d[aug_should_flip, self.joint_mirror_map[k, 1]] = aux

        # Average test time augmentation results
        no_aug = pred3d[2]  # TODO REMOVE DEBUG
        pred3d = pred3d.mean(axis=0)
        bbone_in = bbone_in[int(self.num_aug/2)]
        pred2d = pred2d[int(self.num_aug/2)]
        pred2d_original = pred2d_original[int(self.num_aug/2)]
        is_predicted_to_be_in_fov = is_predicted_to_be_in_fov[int(self.num_aug/2)] if is_predicted_to_be_in_fov is not None else None

        # is_pose_consistent_with_box(pred2d, human)

        return pred3d, edges, bbone_in, pred2d, is_predicted_to_be_in_fov, human, no_aug, pred2d_original


if __name__ == "__main__":
    from utils.params import MetrabsTRTConfig, RealSenseIntrinsics, MainConfig
    from tqdm import tqdm
    from utils.matplotlib_visualizer import MPLPosePrinter
    import matplotlib.pyplot as plt
    import cv2

    args = MainConfig()

    h = HumanPoseEstimator(MetrabsTRTConfig(), RealSenseIntrinsics())
    pose_visualizer = MPLPosePrinter()

    cap = RealSense(width=args.cam_width, height=args.cam_height)
    # cap = cv2.VideoCapture('video.mp4')

    for _ in tqdm(range(10000)):
        ret, img = cap.read()
        pose3d, ed, bbone_input, pose2d, _, _, _, _ = h.estimate(img)
        if pose3d is not None:
            pose3d -= pose3d[0]  # Center

            pose_visualizer.clear()
            pose_visualizer.print_pose(pose3d, ed)
            cv2.imshow("rgb", img)
            plt.pause(0.001)
