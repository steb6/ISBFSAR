import cv2
from ptgaze.gaze_estimator import GazeEstimator
from tqdm import tqdm


class ModelConfig:
    def __init__(self):
        self.name = 'resnet18'


class FaceDetectorConfig:
    def __init__(self):
        self.mode = 'mediapipe'
        self.mediapipe_max_num_faces = 1


class GazeEstimatorConfig:
    def __init__(self):
        self.camera_params = 'assets/camera_params.yaml'
        self.normalized_camera_params = 'assets/eth-xgaze.yaml'
        self.normalized_camera_distance = 0.6
        self.checkpoint = 'modules/focus/focus_detection/modules/raw/eth-xgaze_resnet18.pth'
        self.image_size = [224, 224]


class Config:
    def __init__(self):
        self.face_detector = FaceDetectorConfig()
        self.gaze_estimator = GazeEstimatorConfig()
        self.model = ModelConfig()
        self.mode = 'ETH-XGaze'
        self.device = 'cuda'


cap = cv2.VideoCapture(2)
config = Config()
gaze_estimator = GazeEstimator(config)

for _ in tqdm(range(1000)):

    ok, frame = cap.read()
    if not ok:
        break

    faces = gaze_estimator.detect_faces(frame)

    for face in faces:
        gaze_estimator.estimate_gaze(frame, face)

        v = "{:.2f}".format(face.normalized_gaze_vector[2])
        if face.normalized_gaze_vector[2] < -0.9:
            cv2.putText(frame, v, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, v, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow('frame', frame)
    cv2.waitKey(1)

cap.release()
