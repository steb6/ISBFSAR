from queue import Empty
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

        # Create input-output
        self.debug = debug
        if self.debug:
            from vispy import app
            import multiprocessing
            from threading import Thread
            self.output_queue = multiprocessing.Queue()
            self.input_queue = multiprocessing.Queue()

            def create_visualizer(qi, qo):
                _ = VISPYVisualizer(qi, qo)
                app.run()

            # create_visualizer(self.output_queue, self.input_queue)
            Thread(target=create_visualizer, args=(self.output_queue, self.input_queue)).start()
        else:
            text.start()  # start Client
            self.words_conn = Listener(('localhost', args.just_text_port), authkey=b'secret password').accept()

    def get_frame(self, img=None):
        start = time.time()

        # If img is not given (not a video), try to get img
        if img is None:
            ret, img = self.cap.read()
            if not ret:
                raise Exception("Cannot grab frame!")

        # Estimate 3d skeleton
        pose3d_abs, edges, bbone_in, pose2d_bbone, is_fov, bbox, pose3d_abs_no_aug, pose2d_img = self.hpe.estimate(img)

        # TODO Compute Distance
        # if pose3d_abs is not None:
        #     cam_pos = np.array([0, 0, 0])
        #     man_pose = np.array(pose3d_abs)
        #     d = np.sqrt(np.sum(np.square(cam_pos - man_pose))) / 1000.

        # Normalize

        pose3d_root = pose3d_abs - pose3d_abs[0, :] if pose3d_abs is not None else None
        # pose = pose / self.skeleton_scale  # Normalize  (MetrABS is a cube with sides of 2.2 M)

        # Make inference
        results = self.ar.inference(pose3d_root)

        # Focus
        focus = False
        ret = self.focus.estimate(img)
        if ret is not None:
            focus, face = ret
            img = self.focus.print_bbox(img, face)

        end = time.time()

        # Compute fps
        self.fps_s.append(1. / (end - start))
        fps_s = self.fps_s[-10:]
        fps = sum(fps_s) / len(fps_s)

        if self.debug:

            if pose3d_abs is not None:
                # Print bbox
                x1 = int(bbox[0] * 640)
                y1 = int(bbox[1] * 480)
                x2 = int(bbox[2] * 640)
                y2 = int(bbox[3] * 480)
                # confidence = elem[4]
                # class_id = elem[5]
                img = cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 1)

                # Send to visualizer
                img = cv2.flip(img, 0)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                elements = {"img": img,
                            "pose": pose3d_root,
                            "edges": edges,
                            "fps_without_vis": fps,
                            "focus": focus,
                            "actions": results
                            }
                self.output_queue.put((elements,))

        return img, pose3d_root, results

    def run(self):
        while True:
            # We received a command
            try:
                msg = self.input_queue.get_nowait()
            except Empty:
                # We didn't receive a command, just do inference
                _, _, _ = self.get_frame()
                continue

            msg = msg.strip()
            msg = msg.split()

            # select appropriate command
            if msg[0] == 'close' or msg[0] == 'exit' or msg[0] == 'quit' or msg[0] == 'q':
                break

            elif msg[0] == "add":
                self.learn_command(msg[1:])

            elif msg[0] == "remove":
                self.forget_command(msg[1])

            elif msg[0] == "test":
                self.test_video(msg[1])

            else:
                self.output_queue.put(({"log": "Not a valid command!"},))

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
        # if '.' not in flag:
        requires_focus = "-focus" in flag
        flag = flag[0]
        self.output_queue.put(({"log": "WAIT..."},))
        now = time.time()
        while (time.time() - now) < 3:
            _, _, _ = self.get_frame()
        self.output_queue.put(({"log": "GO!"},))

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
        # else:
        #     if not os.path.exists(flag):
        #         print("Video file does not exists!")
        #         return
        #     # self.cap.release()
        #     video = cv2.VideoCapture(flag)
        #     poses = []
        #     fps = video.get(cv2.CAP_PROP_FPS)
        #     ret, img = video.read()
        #     while ret:
        #         start = time.time()
        #         img = cv2.resize(img, (self.cam_width, self.cam_height))
        #         cv2.waitKey(1)
        #         _, pose, _ = self.get_frame(img)
        #
        #         if pose is not None:
        #             poses.append(pose)
        #
        #         n_skip = int((time.time() - start) * fps)
        #         for _ in range(n_skip):
        #             _, _ = video.read()
        #
        #         ret, img = video.read()
        #
        #     video.release()
        #     self.cap = cv2.VideoCapture(params["cam_id"])
        #     self.cap.set(3, params["cam_width"])
        #     self.cap.set(4, params["cam_height"])
        #
        #     flag = flag.split('/')[1].split('.')[0]  # between / and .

        self.output_queue.put(({"log": "Collected " + str(len(poses)) + " frames"},))

        data = (np.stack(poses), flag, requires_focus)
        self.ar.train(data)


if __name__ == "__main__":

    f = FocusDetector(FocusConfig())
    h = HumanPoseEstimator(MetrabsTRTConfig(), RealSenseIntrinsics())
    n = ActionRecognizer(TRXConfig())
    t = JustText(MainConfig().just_text_port)

    master = ISBFSAR(h, n, t, f, MainConfig(), debug=True)

    master.run()
