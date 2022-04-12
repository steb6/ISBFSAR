from modules.focus.gaze_estimation.focus import FocusDetector
import os
from multiprocessing.connection import Listener
import numpy as np
from tqdm import tqdm
import time
from modules.ar.trx import ActionRecognizer
from utils.input import JustText
import cv2
from playsound import playsound
from utils.input import RealSense
from modules.hpe.hpe import HumanPoseEstimator
from utils.params import MetrabsTRTConfig, RealSenseIntrinsics, MainConfig, FocusConfig
from utils.params import TRXConfig
from utils.vispy_visualizer import VISPYVisualizer


class ISBFSAR:
    def __init__(self, hpe, ar, text, focus, args, debug=True):
        self.hpe = hpe
        self.ar = ar
        self.focus = focus

        # Connect to spt
        text.start()  # start Client
        self.words_conn = Listener(('localhost', args.just_text_port), authkey=b'secret password').accept()

        # Connect to webcam
        if args.cam == "webcam":
            self.cap = cv2.VideoCapture(0)
            self.cap.set(3, args.cam_width)
            self.cap.set(4, args.cam_height)
        elif args.cam == "realsense":
            self.cap = RealSense(width=args.cam_width, height=args.cam_height)
            intrinsics = self.cap.intrinsics()
            i = np.eye(3)
            i[0][0] = intrinsics.fx
            i[0][2] = intrinsics.ppx
            i[1][1] = intrinsics.fy
            i[1][2] = intrinsics.ppy
            self.hpe.intrinsics = i

        self.cam_width = args.cam_width
        self.cam_height = args.cam_height
        self.window_size = args.window_size
        self.fps_s = []
        self.last_poses = []
        self.skeleton_scale = args.skeleton_scale

        # Create visualizer in different thread if required
        self.debug = debug
        if self.debug:
            from vispy import app
            import multiprocessing
            from threading import Thread
            self.queue = multiprocessing.Queue()

            def create_visualizer(qe):
                _ = VISPYVisualizer(qe)
                app.run()

            Thread(target=create_visualizer, args=(self.queue,)).start()

    def get_frame(self, img=None):
        """
        Return the cam  (or frame if provided), estimated skeleton and estimated action
        It also print everything
        """
        start = time.time()

        # If img is not given (not a video), try to get img
        if img is None:
            ret, img = self.cap.read()
            if not ret:
                raise Exception("Cannot grab frame!")

        # Estimate 3d skeleton
        pose3d_abs, edges, bbone_in, pose2d_bbone, is_fov, bbox, pose3d_abs_no_aug, pose2d_img = self.hpe.estimate(img)

        # Compute Distance
        pose3d_root = None
        if pose3d_abs is not None:
            cam_pos = np.array([0, 0, 0])
            man_pose = np.array(pose3d_abs)
            d = np.sqrt(np.sum(np.square(cam_pos - man_pose))) / 1000.

            # Normalize
            pose3d_root = pose3d_abs - pose3d_abs[0, :]  # Center
            pose3d_root_no_aug = pose3d_abs_no_aug - pose3d_abs_no_aug[0]
            # pose = pose / self.skeleton_scale  # Normalize  (MetrABS is a cube with sides of 2.2 M)

        # Make inference
        results = self.ar.inference(pose3d_root)

        # TODO START EXPERIMENT
        # Print movement value
        # if pose is not None:
        #     self.last_poses.append(pose)
        #     if len(self.last_poses) > 1:
        #         self.last_poses = self.last_poses[-self.window_size:]
        #         m = 0
        #         for k in range(len(self.last_poses) - 1):
        #             m += two_poses_movement(self.last_poses[k], self.last_poses[k + 1])
        #         m = m / (len(self.last_poses) - 1)
        # TODO END EXPERIMENT

        # TODO START FOCUS
        ret = self.focus.estimate(img)
        if ret is not None:
            focus, face = ret
            img = self.focus.print_if_is_focus(img)
            img = self.focus.print_bbox(img, face)
        # TODO END FOCUS

        end = time.time()

        # Compute fps
        self.fps_s.append(1. / (end - start))
        fps_s = self.fps_s[-10:]
        fps = sum(fps_s) / len(fps_s)

        if self.debug:
            # Print fps
            cv2.putText(img, "fps: {:.2f}".format(fps), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2,
                        cv2.LINE_AA)

            if pose3d_abs is not None:
                # Print bbox
                x1 = int(bbox[0] * 640)
                y1 = int(bbox[1] * 480)
                x2 = int(bbox[2] * 640)
                y2 = int(bbox[3] * 480)
                # confidence = elem[4]
                # class_id = elem[5]
                img = cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 1)

                # Print pose
                # self.pose_visualizer.print_pose(pose3d_root, edges, 'g')
                self.queue.put((pose3d_root, edges, img))

                # Print distance
                if pose3d_abs is not None:
                    cv2.putText(img, "Dist (M): {:.2f}".format(d), (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2,
                                cv2.LINE_AA)

                # Print action
                i = 0
                if results is not None:
                    m = max(results.values())
                    for r in results.keys():
                        if results[r] == m:
                            color = (0, 255, 0)
                        else:
                            color = (255, 0, 0)
                        cv2.putText(img, r + ": " + '%.2f' % results[r], (50, 200 + i * 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, color,
                                    2,
                                    cv2.LINE_AA)
                        i += 1

                # cv2.imshow("2D predictions", bbone_in.astype(np.uint8))

            # cv2.imshow("Cam", img)
            # cv2.waitKey(1)

        return img, pose3d_root, results

    def run(self):
        # while True:
        for _ in tqdm(range(1000)):
            # We received a command
            if self.words_conn.poll():
                # Get msg
                msg = self.words_conn.recv().strip()
                msg = msg.split()

                # select appropriate command
                if msg[0] == 'close' or msg[0] == 'exit' or msg[0] == 'quit' or msg[0] == 'q':
                    break

                elif msg[0] == "add":
                    action = msg[1]
                    self.learn_command(action)

                elif msg[0] == "remove":
                    self.forget_command(msg[1])

                elif msg[0] == "test":
                    self.test_video(msg[1])

                elif msg[0] == "debug":
                    print("### VISUALIZING DATASET ###")
                    for elem in self.ar.debug():
                        print(elem[1])
                        pose = elem[0].reshape(16, 30, 3)*self.skeleton_scale
                        for p in pose:
                            self.pose_visualizer.print_pose(p, elem[2])
                            time.sleep(0.2)
                        time.sleep(1)

                else:
                    print("Not a valid command!")
                    print("Valid commands are 'add', 'remove' and 'quit'")
                    print("If a video is provided, add the extension in order to load it")
            else:
                # We didn't receive a command, just do inference
                _, _, _ = self.get_frame()

        # clean
        self.words_conn.close()

    def test_video(self, path):
        if not os.path.exists(path):
            print("Video file does not exists!")
            return
        # self.cap.release()
        video = cv2.VideoCapture(path)
        i = 0
        fps = video.get(cv2.CAP_PROP_FPS)
        ret, img = video.read()
        while ret:
            start = time.time()
            key = cv2.waitKey(1)
            if key > -1:
                break
            print(i)
            i += 1
            _, _, _ = self.get_frame(img)

            n_skip = int((time.time() - start) * fps)
            for _ in range(n_skip):
                _, _ = video.read()

            ret, img = video.read()

        video.release()
        # self.cap = cv2.VideoCapture(params["cam_id"])
        # self.cap.set(3, params["cam_width"])
        # self.cap.set(4, params["cam_height"])

    def forget_command(self, flag):
        self.ar.remove(flag)

    def learn_command(self, flag):
        # If a string is provided
        if '.' not in flag:
            print("WAIT")
            now = time.time()
            while (time.time() - now) < 3:
                _, _, _ = self.get_frame()
            print("GO")

            playsound('assets' + os.sep + 'start.wav')
            poses = []
            with tqdm(total=self.window_size, position=0, leave=True) as progress_bar:
                while len(poses) < self.window_size:
                    _, pose, _ = self.get_frame()
                    if pose is not None:
                        poses.append(pose)
                        progress_bar.update()
            playsound('assets' + os.sep + 'stop.wav')
        # If a path to a video is provided
        else:
            if not os.path.exists(flag):
                print("Video file does not exists!")
                return
            # self.cap.release()
            video = cv2.VideoCapture(flag)
            poses = []
            fps = video.get(cv2.CAP_PROP_FPS)
            ret, img = video.read()
            while ret:
                start = time.time()
                img = cv2.resize(img, (self.cam_width, self.cam_height))
                cv2.waitKey(1)
                _, pose, _ = self.get_frame(img)

                if pose is not None:
                    poses.append(pose)

                n_skip = int((time.time() - start) * fps)
                for _ in range(n_skip):
                    _, _ = video.read()

                ret, img = video.read()

            video.release()
            # self.cap = cv2.VideoCapture(params["cam_id"])
            # self.cap.set(3, params["cam_width"])
            # self.cap.set(4, params["cam_height"])

            flag = flag.split('/')[1].split('.')[0]  # between / and .

        print("Collected " + str(len(poses)) + " frames")

        data = (np.stack(poses), flag)
        self.ar.train(data)


if __name__ == "__main__":

    f = FocusDetector(FocusConfig())
    h = HumanPoseEstimator(MetrabsTRTConfig(), RealSenseIntrinsics())
    n = ActionRecognizer(TRXConfig())
    t = JustText(MainConfig().just_text_port)

    master = ISBFSAR(h, n, t, f, MainConfig(), debug=True)

    master.run()
