import platform

ubuntu = platform.system() == "Linux"
engine_dir = "engines"
print("Ubuntu: {}".format(ubuntu))

seq_len = 16


class MainConfig(object):
    def __init__(self):
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
        self.optimize_every = 1
        self.batch_size = 32
        self.start_discriminator_after_epoch = -1
        self.first_mile = 15
        self.second_mile = 1500
        self.model = "DISC"  # DISC or EXP
        self.final_ckpt_path = "modules/ar/modules/raws/DISC.pth"
        # CHOICE DATASET
        self.exemplars_path = "D:\\datasets\\metrabs_trx_skeletons_exemplars" if not ubuntu else "../metrabs_trx_skeletons_exemplars"
        self.data_path = "D:/datasets/nturgbd_metrabs_2/" if not ubuntu else "../nturgbd_metrabs_2/"
        self.n_joints = 30
        # self.data_path = "D:/datasets/nturgbd_trx_skeletons_ALL" if not ubuntu else "../nturgbd_trx_skeletons_ALL"
        # self.n_joints = 25
        # END CHOICE DATASET
        self.n_workers = 2 if not ubuntu else 20
        self.n_epochs = 1000
        self.log_every = 10 if not ubuntu else 1000
        self.trt_path = 'modules/ar/modules/{}/trx.engine'.format(engine_dir)
        self.trans_linear_in_dim = 256
        self.trans_linear_out_dim = 128
        self.way = 12
        self.shot = 1
        self.query_per_class = 1
        self.trans_dropout = 0.
        self.seq_len = seq_len
        self.num_gpus = 1
        self.temp_set = [2]
        self.device = 'cuda'


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
        self.focus_model = 'modules/focus/mutual_gaze/focus_detection/checkpoints/group_RNET MARIA SMALL_sess_4_f1_0.82.pth'

        self.augmentation_size = 0.4
        self.dataset = "focus_dataset_small_easy"
        self.model = "rnet"

        self.batch_size = 32
        self.lr = 1e-6
        self.log_every = 10
        self.pretrained = True
        self.n_epochs = 1000
        self.patience = 100
