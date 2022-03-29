import os
import numpy as np
import cv2
from scipy.spatial.transform import Rotation
from tqdm import tqdm
from modules.focus.focus import FocusDetector, convert_pt
from utils.params import FocusConfig

if __name__ == "__main__":
    import numpy as np
    import yaml
    cap = cv2.VideoCapture('video.mp4')
    det = FocusDetector(FocusConfig())

    area_thr = 0.03  # head bounding box must be over this value to be close
    close_thr = -0.95  # When close, z value over this thr is considered focus
    dist_thr = 0.2  # when distant, roll under this thr is considered focus
    foc_rot_thr = 0.7  # when close, roll above this thr is considered not focus
    patience = 16  # result is based on the majority of previous observations

    with open('C:/Users/sberti/AppData/Local/mambaforge/envs/robotologyenv/Lib/'
              'site-packages/ptgaze/data/calib/sample_params.yaml') as f:
        data = yaml.safe_load(f)
    camera_matrix = np.array(data['camera_matrix']['data']).reshape(3, 3)

    with open('D:/datasets/mutualGaze_dataset/realsense/eyecontact_annotations_all.txt', "r") as infile:
        files = infile.readlines()
    for file in files:
        file, label = file.split(' ')
        file = 'D:/datasets/mutualGaze_dataset/realsense' + file[1:]
        frame = cv2.imread(file)
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
                # head_pose = np.linalg.inv(face.normalizing_rot.as_matrix()) @ face.head_pose_rot.as_rotvec()
                head_pose = face.head_pose_rot.as_rotvec()
                axes3d = axes3d * 0.05  # length
                points2d, _ = cv2.projectPoints(axes3d, head_pose, face.head_position,
                                                camera_matrix,
                                                np.array([[0.], [0.], [0.], [0.], [0.]]))
                axes2d = points2d.reshape(-1, 2)
                center = face.landmarks[1]  # center point index
                center = convert_pt(center)
                for pt, color in zip(axes2d, [(0, 0, 255), (0, 255, 0), (255, 0, 0)]):
                    pt = convert_pt(pt)
                    cv2.line(frame, center, pt, color, 2, cv2.LINE_AA)  # lw

                # Print focus
                score1 = abs(head_pose[1])  # on axis x
                score2 = abs(head_pose[0])  # on axis y
                cv2.putText(frame, "{:.2f} < {:.2f}, {:.2f} < {:.2f}".format(score1, dist_thr, score2, dist_thr), (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 0, 0), 2, cv2.LINE_AA)
                focus = score1 < dist_thr and score2 < dist_thr

            # Print focus
            if focus:
                cv2.putText(frame, "FOCUS", (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            else:
                cv2.putText(frame, "NOT FOCUS", (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

            if bool(int(label[0])):
                cv2.putText(frame, "REAL: FOCUS", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            else:
                cv2.putText(frame, "REAL: NOT FOCUS", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

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
        cv2.waitKey(0)

    cap.release()
