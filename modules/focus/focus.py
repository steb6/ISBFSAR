import cv2
from ptgaze.gaze_estimator import GazeEstimator
from tqdm import tqdm
from scipy.spatial.transform import Rotation


class FocusDetector:
    def __init__(self):

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
                self.checkpoint = 'modules/focus/modules/raw/eth-xgaze_resnet18.pth'
                self.image_size = [224, 224]

        class Config:
            def __init__(self):
                self.face_detector = FaceDetectorConfig()
                self.gaze_estimator = GazeEstimatorConfig()
                self.model = ModelConfig()
                self.mode = 'ETH-XGaze'
                self.device = 'cuda'

        config = Config()
        self.gaze_estimator = GazeEstimator(config)

    def estimate(self, img):

        faces = self.gaze_estimator.detect_faces(img)

        if len(faces) == 0:
            return None

        fc = faces[0]  # We can only have one face
        self.gaze_estimator.estimate_gaze(img, fc)
        return fc


def convert_pt(point):
    return tuple(np.round(point).astype(int).tolist())


if __name__ == "__main__":
    import numpy as np
    import yaml
    cap = cv2.VideoCapture('video.mp4')
    det = FocusDetector()

    area_thr = 0.03  # head bounding box must be over this value to be close
    close_thr = -0.95  # When close, z value over this thr is considered focus
    dist_thr = 0.2  # when distant, roll under this thr is considered focus
    foc_rot_thr = 0.7  # when close, roll above this thr is considered not focus
    patience = 16  # result is based on the majority of previous observations

    with open('C:/Users/sberti/AppData/Local/mambaforge/envs/robotologyenv/Lib/'
              'site-packages/ptgaze/data/calib/sample_params.yaml') as f:
        data = yaml.safe_load(f)
    camera_matrix = np.array(data['camera_matrix']['data']).reshape(3, 3)
    focuses = []
    for _ in tqdm(range(100000)):

        ok, frame = cap.read()
        face = det.estimate(frame)
        if face is not None:

            cv2.imshow("NORMALIZED", face.normalized_image)

            # Print bounding box and compute area
            bbox = np.round(face.bbox).astype(int).tolist()
            frame = cv2.rectangle(frame, tuple(bbox[0]), tuple(bbox[1]), (255, 0, 0), 1)
            area = ((bbox[1][0] - bbox[0][0]) * (bbox[1][1] - bbox[0][1])) / (640 * 480)
            cv2.putText(frame, "Area: {:.2f} > {:.2f}".format(area, area_thr), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 0), 2, cv2.LINE_AA)

            # HUMAN IS CLOSE, USE EYES
            focus = None
            if area > area_thr:
                cv2.putText(frame, "CLOSE", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                            cv2.LINE_AA)

                # Print gaze pose
                axes3d = np.eye(3, dtype=float) @ Rotation.from_euler(
                    'XYZ', [0, np.pi, 0]).as_matrix()
                axes3d = axes3d * 0.05  # length
                points2d, _ = cv2.projectPoints(axes3d, face.gaze_vector, face.head_position,
                                                camera_matrix,
                                                np.array([[0.], [0.], [0.], [0.], [0.]]))
                axes2d = points2d.reshape(-1, 2)
                center = face.landmarks[1]  # center point index
                center = convert_pt(center)
                for pt, color in zip(axes2d, [(0, 0, 255), (0, 255, 0), (255, 0, 0)]):
                    pt = convert_pt(pt)
                    cv2.line(frame, center, pt, color, 2, cv2.LINE_AA)  # lw

                # Print focus
                score = face.normalized_gaze_vector[2]
                score_rot = abs(face.head_pose_rot.as_rotvec()[1])
                cv2.putText(frame, "{:.2f} < {:.2f} and {:.2f} < {:.2f}".format(score, close_thr, score_rot, foc_rot_thr), (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 0, 0), 2, cv2.LINE_AA)
                focus = score < close_thr and score_rot < foc_rot_thr

            # HUMAN IS NOT CLOSE, USE HEAD POSE
            else:
                cv2.putText(frame, "NOT CLOSE", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                            cv2.LINE_AA)

                # Print head pose
                axes3d = np.eye(3, dtype=float) @ Rotation.from_euler(
                    'XYZ', [0, np.pi, 0]).as_matrix()
                axes3d = np.linalg.inv(face.normalizing_rot.as_matrix()) @ axes3d  # TODO TEST
                axes3d = axes3d * 0.05  # length
                points2d, _ = cv2.projectPoints(axes3d, face.head_pose_rot.as_rotvec(), face.head_position,
                                                camera_matrix,
                                                np.array([[0.], [0.], [0.], [0.], [0.]]))
                axes2d = points2d.reshape(-1, 2)
                center = face.landmarks[1]  # center point index
                center = convert_pt(center)
                for pt, color in zip(axes2d, [(0, 0, 255), (0, 255, 0), (255, 0, 0)]):
                    pt = convert_pt(pt)
                    cv2.line(frame, center, pt, color, 2, cv2.LINE_AA)  # lw

                # Print focus
                head_pose = face.head_pose_rot.as_rotvec()
                score = abs(head_pose[1])
                cv2.putText(frame, "{:.2f} < {:.2f}".format(score, dist_thr), (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 0, 0), 2, cv2.LINE_AA)
                focus = score < dist_thr

            # Print focus
            focuses.append(focus)
            focuses = focuses[-16:]

            focus = focuses.count(True) > len(focuses) / 2

            if focus:
                cv2.putText(frame, "FOCUS", (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            else:
                cv2.putText(frame, "NOT FOCUS", (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

            # # Print gaze vector
            # point0 = face.center
            # point1 = face.center + 0.05 * face.gaze_vector
            # points3d = np.vstack([point0, point1])
            # points2d, _ = cv2.projectPoints(points3d, np.zeros(3, dtype=float), np.zeros(3, dtype=float),
            #                                 camera_matrix,
            #                                 np.array([[0.], [0.], [0.], [0.], [0.]]))
            # axes2d = points2d.reshape(-1, 2)
            # pt0 = convert_pt(axes2d[0])
            # pt1 = convert_pt(axes2d[1])
            # frame = cv2.line(frame, pt0, pt1, (255, 0, 0), 1, cv2.LINE_AA)

        cv2.imshow('frame', frame)
        cv2.waitKey(100)

    cap.release()
