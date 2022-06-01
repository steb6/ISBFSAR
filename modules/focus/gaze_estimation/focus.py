import cv2
from modules.focus.gaze_estimation.pytorch_mpiigaze_demo.ptgaze.gaze_estimator import GazeEstimator
from tqdm import tqdm
from scipy.spatial.transform import Rotation
import yaml
import numpy as np


class FocusDetector:
    def __init__(self, config):
        self.gaze_estimator = GazeEstimator(config)

        self.area_thr = config.area_thr  # head bounding box must be over this value to be close
        self.close_thr = config.close_thr  # When close, z value over this thr is considered focus
        self.dist_thr = config.dist_thr  # when distant, roll under this thr is considered focus
        self.foc_rot_thr = config.foc_rot_thr  # when close, roll above this thr is considered not focus
        self.patience = config.patience  # result is based on the majority of previous observations

        self.is_close = None
        self.is_focus = None

        with open(config.sample_params_path) as f:
            self.data = yaml.safe_load(f)
        self.camera_matrix = np.array(self.data['camera_matrix']['data']).reshape(3, 3)
        self.focuses = []

    def print_bbox(self, frame, face):
        bbox = np.round(face.bbox).astype(int).tolist()
        frame = cv2.rectangle(frame, tuple(bbox[0]), tuple(bbox[1]), (255, 0, 0), 1)
        return frame

    def print_close_or_not(self, frame):
        if self.is_close is not None:
            if not self.is_close:
                frame = cv2.putText(frame, "NOT CLOSE", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                                    cv2.LINE_AA)
            else:
                frame = cv2.putText(frame, "CLOSE", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                                    cv2.LINE_AA)
        return frame

    def print_bbox_area(self, frame, face):
        bbox = face.bbox
        area = ((bbox[1][0] - bbox[0][0]) * (bbox[1][1] - bbox[0][1])) / (640 * 480)
        frame = cv2.putText(frame, "Area: {:.2f} > {:.2f}".format(area, self.area_thr), (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 0, 0), 2, cv2.LINE_AA)
        return frame

    def print_gaze_pose(self, frame, face):
        # Print gaze pose
        axes3d = np.eye(3, dtype=float) @ Rotation.from_euler(
            'XYZ', [0, np.pi, 0]).as_matrix()
        axes3d = axes3d * 0.05  # length
        points2d, _ = cv2.projectPoints(axes3d, face.gaze_vector, face.head_position,
                                        self.camera_matrix,
                                        np.array([[0.], [0.], [0.], [0.], [0.]]))
        axes2d = points2d.reshape(-1, 2)
        center = face.landmarks[1]  # center point index
        center = convert_pt(center)
        for pt, color in zip(axes2d, [(0, 0, 255), (0, 255, 0), (255, 0, 0)]):
            pt = convert_pt(pt)
            frame = cv2.line(frame, center, pt, color, 2, cv2.LINE_AA)  # lw
        return frame

    def print_head_pose(self, frame, face):
        # Print head pose
        axes3d = np.eye(3, dtype=float) @ Rotation.from_euler(
            'XYZ', [0, np.pi, 0]).as_matrix()
        head_pose = face.head_pose_rot.as_rotvec()  # TODO BEFORE
        # head_pose = np.linalg.inv(face.normalizing_rot.as_matrix()) @ face.head_pose_rot.as_rotvec()  # TODO TRY
        axes3d = axes3d * 0.05  # length
        points2d, _ = cv2.projectPoints(axes3d, head_pose, face.head_position,
                                        self.camera_matrix,
                                        np.array([[0.], [0.], [0.], [0.], [0.]]))
        axes2d = points2d.reshape(-1, 2)
        center = face.landmarks[1]  # center point index
        center = convert_pt(center)
        for pt, color in zip(axes2d, [(0, 0, 255), (0, 255, 0), (255, 0, 0)]):
            pt = convert_pt(pt)
            frame = cv2.line(frame, center, pt, color, 2, cv2.LINE_AA)  # lw
        return frame

    def print_if_is_focus(self, frame):
        if self.is_focus:
            frame = cv2.putText(frame, "FOCUS", (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            frame = cv2.putText(frame, "NOT FOCUS", (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        return frame

    def print_score(self, frame, face):
        if self.is_close:
            score = face.normalized_gaze_vector[2]
            score_rot = abs(face.head_pose_rot.as_rotvec()[1])
            frame = cv2.putText(frame, "{:.2f} < {:.2f} and {:.2f} < {:.2f}".format(score, self.close_thr,
                                                                                    score_rot, self.foc_rot_thr),
                                (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
        else:
            head_pose = face.normalized_head_rot2d
            score = abs(head_pose[1])
            frame = cv2.putText(frame, "{:.2f} < {:.2f}".format(score, self.dist_thr), (10, 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
        return frame

    def estimate(self, frame):
        faces = self.gaze_estimator.detect_faces(frame)

        if len(faces) == 0:
            return None

        fc = faces[0]  # We can only have one face
        self.gaze_estimator.estimate_gaze(frame, fc)

        face = fc
        focus = None
        if face is not None:

            bbox = face.bbox
            area = ((bbox[1][0] - bbox[0][0]) * (bbox[1][1] - bbox[0][1])) / (640 * 480)

            # HUMAN IS CLOSE, USE EYES
            if area > self.area_thr:
                self.is_close = True
                score = face.normalized_gaze_vector[2]
                score_rot = abs(face.head_pose_rot.as_rotvec()[1])
                focus = score < self.close_thr and score_rot < self.foc_rot_thr

            # HUMAN IS NOT CLOSE, USE HEAD POSE
            else:
                self.is_close = False
                head_pose = face.normalized_head_rot2d
                score = abs(head_pose[1])
                focus = score < self.dist_thr

            # Print focus
            self.focuses.append(focus)
            self.focuses = self.focuses[-self.patience:]
            self.is_focus = self.focuses.count(True) > len(self.focuses) / 2

        return focus, fc


def convert_pt(point):
    return tuple(np.round(point).astype(int).tolist())


if __name__ == "__main__":
    from utils.params import FocusConfig

    cap = cv2.VideoCapture(0)
    # ok, img = cap.read()
    # img = cv2.imread("frame.jpg")
    det = FocusDetector(FocusConfig())

    for _ in tqdm(range(10000000)):
        ok, img = cap.read()
        f = det.estimate(img)

        if f is not None:
            _, f = f

            # f.head_pose_rot.head_pose = f.head_pose_rot.as_rotvec() @ f.normalizing_rot

            img = det.print_close_or_not(img)
            img = det.print_bbox_area(img, f)
            img = det.print_if_is_focus(img)
            img = det.print_score(img, f)
            img = det.print_bbox(img, f)

            if det.is_close:
                img = det.print_gaze_pose(img, f)
            else:
                img = det.print_head_pose(img, f)

            cv2.imshow('normalized', f.normalized_image)
        cv2.imshow('frame', img)
        cv2.waitKey(1)

    cap.release()
