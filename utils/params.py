seq_len = 16
n_joints = 30


class MainConfig(object):
    def __init__(self):
        self.cam = "realsense"
        self.ar = "trx"
        self.cam_width = 640
        self.cam_height = 480
        self.window_size = seq_len
        self.just_text_port = 6050
        self.skeleton_scale = 2200.


class MetrabsTRTConfig(object):
    def __init__(self):
        self.yolo_engine_path = 'modules/hpe/modules/engines/yolo.engine'
        self.image_transformation_path = 'modules/hpe/modules/engines/image_transformation.engine'
        self.bbone_engine_path = 'modules/hpe/modules/engines/bbone.engine'
        self.head_weight_path = 'modules/hpe/modules/numpy/head_weights.npy'
        self.head_bias_path = 'modules/hpe/modules/numpy/head_bias.npy'
        self.expand_joints_path = 'assets/32_to_122.npy'
        self.skeleton_types_path = 'assets/skeleton_types.pkl'
        self.skeleton = 'smpl+head_30'
        self.yolo_thresh = 0.3
        self.nms_thresh = 0.7
        self.num_aug = 5


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
        self.model_path = 'modules/ar/trx/modules/raws/671.pth'
        self.trt_path = 'modules/ar/trx/modules/engines/fix_trx.engine'
        self.trans_linear_in_dim = 256
        self.trans_linear_out_dim = 128
        self.way = 5
        self.shot = 1
        self.query_per_class = 1
        self.trans_dropout = 0.
        self.seq_len = seq_len
        self.n_joints = n_joints
        self.num_gpus = 1
        self.temp_set = [2]
        self.device = 'cuda'


class LSTMTrainConfig(object):
    def __init__(self):
        self.hidden_size = 128
        self.n_epochs = 5
        self.augmentation_factor = 100
        self.augmentation_noise = 0.01
        self.validation_size = 0.3
        self.patience = 5
        self.seq_len = seq_len
        self.device = 'cuda'
        self.n_joints = n_joints
