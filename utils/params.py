import platform
import os

input_type = "skeleton"  # rgb, skeleton or hybrid
skeleton_type = 'smpl+head_30'

docker = os.environ.get('AM_I_IN_A_DOCKER_CONTAINER', False)
seq_len = 8 if input_type != "skeleton" else 16
ubuntu = platform.system() == "Linux"
engine_dir = "engines" if not docker else os.path.join("engines", "docker")
print("Ubuntu: {}".format(ubuntu))


class MainConfig(object):
    def __init__(self):
        self.input_type = input_type  # rgb or skeleton
        self.cam = "realsense"  # webcam or realsense
        self.cam_width = 640
        self.cam_height = 480
        self.window_size = seq_len
        self.skeleton_scale = 2200.
        self.acquisition_time = 3  # Seconds


class MetrabsTRTConfig(object):
    def __init__(self):
        self.yolo_engine_path = os.path.join('modules', 'hpe', 'weights', engine_dir, 'yolo.engine')
        self.image_transformation_path = os.path.join('modules', 'hpe', 'weights', engine_dir, 'image_transformation1.engine')
        self.bbone_engine_path = os.path.join('modules', 'hpe', 'weights', engine_dir, 'bbone1.engine')
        self.heads_engine_path = os.path.join('modules', 'hpe', 'weights', engine_dir, 'heads1.engine')
        self.expand_joints_path = 'assets/32_to_122.npy'
        self.skeleton_types_path = 'assets/skeleton_types.pkl'
        self.skeleton = skeleton_type
        self.yolo_thresh = 0.3
        self.nms_thresh = 0.7
        self.num_aug = 0  # if zero, disables test time augmentation
        self.just_box = input_type == "rgb"


class RealSenseIntrinsics(object):
    def __init__(self):
        self.fx = 384.025146484375
        self.fy = 384.025146484375
        self.ppx = 319.09661865234375
        self.ppy = 237.75723266601562
        self.width = 640
        self.height = 480


class TRXConfig(object):
    def __init__(self):
        # MAIN
        self.model = "DISC"  # DISC or EXP
        self.input_type = input_type  # skeleton or rgb
        self.way = 5
        self.shot = 1
        self.device = 'cuda'
        self.skeleton_type = skeleton_type

        # CHOICE DATASET
        data_name = "NTURGBD_to_YOLO_METRO_122"
        self.data_path = f"D:\\datasets\\{data_name}" if not ubuntu else f"../datasets/{data_name}"
        self.n_joints = 30

        # TRAINING
        self.initial_lr = 1e-2 if self.input_type == "skeleton" else 3e-4
        self.n_task = (100 if self.input_type == "skeleton" else 30) if not ubuntu else (10000 if self.input_type == "skeleton" else 500)
        self.optimize_every = 1  # Put to 1 if not used, not 0 or -1!
        self.batch_size = 1 if not ubuntu else (32 if self.input_type == "skeleton" else 4)
        self.n_epochs = 10000
        self.start_discriminator_after_epoch = 0  # self.n_epochs  # TODO CAREFUL
        self.first_mile = self.n_epochs  # 15 TODO CAREFUL
        self.second_mile = self.n_epochs  # 1500 TODO CAREFUL
        self.n_workers = 0 if not ubuntu else 16
        self.log_every = 10 if not ubuntu else 1000
        self.eval_every_n_epoch = 10

        # MODEL
        self.trans_linear_in_dim = 256 if self.input_type == "skeleton" else 1000 if self.input_type == "rgb" else 512
        self.trans_linear_out_dim = 128
        self.query_per_class = 1
        self.trans_dropout = 0.
        self.num_gpus = 4
        self.temp_set = [2]
        self.checkpoints_path = "checkpoints"

        # DEPLOYMENT
        if input_type == "rgb":
            self.final_ckpt_path = "modules/ar/modules/raws/rgb/3000.pth"
        elif input_type == "skeleton":
            self.final_ckpt_path = "modules/ar/modules/raws/DISC.pth"
        elif input_type == "hybrid":
            self.final_ckpt_path = "modules/ar/modules/raws/hybrid/1714_truncated_resnet.pth"
        self.trt_path = 'modules/ar/modules/{}/trx.engine'.format(engine_dir)
        self.seq_len = seq_len


class FocusModelConfig:
    def __init__(self):
        self.name = 'resnet18'


class FaceDetectorConfig:
    def __init__(self):
        self.mode = 'mediapipe'
        self.mediapipe_max_num_faces = 1
        self.mediapipe_static_image_mode = False


class GazeEstimatorConfig:
    def __init__(self):
        self.camera_params = 'assets/camera_params.yaml'
        self.normalized_camera_params = 'assets/eth-xgaze.yaml'
        self.normalized_camera_distance = 0.6
        self.checkpoint = 'modules/focus/gaze_estimation/modules/raw/eth-xgaze_resnet18.pth'
        self.image_size = [224, 224]


class FocusConfig:
    def __init__(self):
        # GAZE ESTIMATION
        self.face_detector = FaceDetectorConfig()
        self.gaze_estimator = GazeEstimatorConfig()
        self.model = FocusModelConfig()
        self.mode = 'ETH-XGaze'
        self.device = 'cuda'
        self.area_thr = 0.03  # head bounding box must be over this value to be close
        self.close_thr = -0.95  # When close, z value over this thr is considered focus
        self.dist_thr = 0.3  # when distant, roll under this thr is considered focus
        self.foc_rot_thr = 0.7  # when close, roll above this thr is considered not focus
        self.patience = 3  # result is based on the majority of previous observations
        self.sample_params_path = "assets/sample_params.yaml"


class MutualGazeConfig:
    def __init__(self):
        self.data_path = 'D:/datasets/mutualGaze_dataset' if not ubuntu else "/home/IIT.LOCAL/sberti/datasets/mutualGaze_dataset"
        self.head_model = 'modules/focus/mutual_gaze/head_detection/epoch_0.pth'
        self.focus_model = 'modules/focus/mutual_gaze/focus_detection/checkpoints/sess_0_f1_1.00.pth'
        self.ckpts_path = 'modules/focus/mutual_gaze/focus_detection/checkpoints'

        self.augmentation_size = 0.8
        self.dataset = "focus_dataset_heads"
        self.model = "facenet"  # facenet, resnet

        self.batch_size = 8
        self.lr = 1e-6
        self.log_every = 10
        self.pretrained = True
        self.n_epochs = 1000
