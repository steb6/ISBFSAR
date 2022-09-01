import platform

input_type = "rgb"  # rgb or skeleton
seq_len = 8 if input_type == "rgb" else 16  # 8 for rgb and 16 for skeleton


ubuntu = platform.system() == "Linux"
engine_dir = "engines"
print("Ubuntu: {}".format(ubuntu))


class MainConfig(object):
    def __init__(self):
        self.input_type = input_type  # rgb or skeleton
        self.cam = "realsense"  # webcam or realsense
        self.cam_width = 640
        self.cam_height = 480
        self.window_size = seq_len
        self.skeleton_scale = 2200.


class MetrabsTRTConfig(object):
    def __init__(self):
        self.yolo_engine_path = 'modules/hpe/modules/{}/yolo.engine'.format(engine_dir)
        self.image_transformation_path = 'modules/hpe/modules/{}/image_transformation1.engine'.format(engine_dir)
        self.bbone_engine_path = 'modules/hpe/modules/{}/bbone1.engine'.format(engine_dir)
        self.heads_engine_path = 'modules/hpe/modules/{}/heads1.engine'.format(engine_dir)
        self.expand_joints_path = 'assets/32_to_122.npy'
        self.skeleton_types_path = 'assets/skeleton_types.pkl'
        self.skeleton = 'smpl+head_30'
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

        # CHOICE DATASET
        # NTU METRABS SKELETONS
        # self.exemplars_path = "D:\\datasets\\metrabs_trx_skeletons_exemplars" if not ubuntu else "../metrabs_trx_skeletons_exemplars"
        # self.data_path = "D:/datasets/nturgbd_metrabs_2/" if not ubuntu else "../nturgbd_metrabs_2/"
        # self.n_joints = 30
        # NTU ORIGINAL SKELETONS
        # self.data_path = "D:/datasets/nturgbd_trx_skeletons_ALL" if not ubuntu else "../nturgbd_trx_skeletons_ALL"
        # self.n_joints = 25
        # NTU RGB IMAGES
        self.data_path = "D:/datasets/NTURGBD_FINAL_IMAGES" if not ubuntu else "../NTURGBD_FINAL_IMAGES_no_pad"

        # TRAINING
        self.initial_lr = 1e-2 if self.input_type == "skeleton" else 1e-3
        self.n_task = (100 if self.input_type == "skeleton" else 30) if not ubuntu else (10000 if self.input_type == "skeleton" else 100)
        self.optimize_every = 16  # Put to 1 if not used, not 0 or -1!
        self.batch_size = 1 if not ubuntu else (32 if self.input_type == "skeleton" else 4)
        self.n_epochs = 10000
        self.start_discriminator_after_epoch = 0  # self.n_epochs  # TODO CAREFUL
        self.first_mile = self.n_epochs  # 15 TODO CAREFUL
        self.second_mile = self.n_epochs  # 1500 TODO CAREFUL
        self.n_workers = 0 if not ubuntu else 20

        self.log_every = 10 if not ubuntu else 1000
        self.trans_linear_in_dim = 256 if self.input_type == "skeleton" else 1000
        self.trans_linear_out_dim = 128
        self.query_per_class = 1
        self.trans_dropout = 0.
        self.num_gpus = 4
        self.temp_set = [2]
        self.checkpoints_path = "checkpoints"

        # DEPLOYMENT
        self.final_ckpt_path = "modules/ar/modules/raws/DISC.pth" if self.input_type == "skeleton" else "modules/ar/modules/raws/rgb/3000.pth"
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
        self.head_model = 'modules/focus/mutual_gaze/head_detection/epoch_0.pth'
        self.focus_model = 'modules/focus/mutual_gaze/focus_detection/checkpoints/sess_0_f1_1.00.pth'

        self.augmentation_size = 0.8
        self.dataset = "focus_dataset_heads"
        self.model = "facenet"  # facenet, resnet

        self.batch_size = 32
        self.lr = 1e-6
        self.log_every = 10
        self.pretrained = True
        self.n_epochs = 1000
        self.patience = 100
