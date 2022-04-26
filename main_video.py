from modules.focus.gaze_estimation.focus import FakeFocusDetector as FocusDetector
# from modules.focus.mutual_gaze.focus import FocusDetector
import os
import numpy as np
from tqdm import tqdm
import time
from modules.ar.trx import ActionRecognizer
import cv2
from playsound import playsound
from utils.input import just_text
from modules.hpe.hpe import HumanPoseEstimator
from utils.params import MetrabsTRTConfig, RealSenseIntrinsics, MainConfig, FocusConfig
from utils.params import TRXConfig
from utils.output import VISPYVisualizer
import multiprocessing
from threading import Thread


class ISBFSAR:
    def __init__(self, hpe, ar, focus, args, debug=False):
        self.hpe = hpe
        self.ar = ar
        self.focus = focus
        self.is_running = True

        # Connect to webcam
        self.cap = cv2.VideoCapture('assets/test_gaze_no_mask.mp4')

        self.cam_width = args.cam_width
        self.cam_height = args.cam_height
        self.window_size = args.window_size
        self.fps_s = []
        self.last_poses = []
        self.skeleton_scale = args.skeleton_scale

        # Create input
        self.input_queue = multiprocessing.Queue()
        self.input_thread = Thread(target=just_text, args=(self.input_queue, lambda: self.is_running))
        self.input_thread.start()

        # Create output
        self.debug = debug
        if self.debug:
            self.output_queue = multiprocessing.Queue()
            self.output_thread = Thread(target=VISPYVisualizer.create_visualizer,
                                        args=(self.output_queue, self.input_queue, lambda: self.is_running))
            self.output_thread.start()

    def get_frame(self, img=None):
        start = time.time()

        # If img is not given (not a video), try to get img
        if img is None:
            ret, img = self.cap.read()
            if not ret:
                raise Exception("Cannot grab frame!")

        # Estimate 3d skeleton
        pose3d_abs, edges, bbox = self.hpe.estimate(img)

        # Compute distance
        d = None
        if pose3d_abs is not None:
            cam_pos = np.array([0, 0, 0])
            man_pose = np.array(pose3d_abs[0])
            d = np.sqrt(np.sum(np.square(cam_pos - man_pose)))

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
                x1, x2, y1, y2 = bbox
                img = cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 1)

                # Send to visualizer
                img = cv2.flip(img, 0)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                elements = {"img": img,
                            "pose": pose3d_root,
                            "edges": edges,
                            "fps": fps,
                            "focus": focus,
                            "actions": results,
                            "distance": d * 2  # TODO fix
                            }
                self.output_queue.put((elements,))

        return img, pose3d_root, results

    def log(self, msg):
        self.output_queue.put(({"log": msg},))
        print(msg)

    def run(self):
        # while True:
        for _ in tqdm(list(range(10000))):
            # We received a command
            if not self.input_queue.empty():
                msg = self.input_queue.get()
            else:
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
                self.log("Not a valid command!")

        # clean  # TODO TERMINATE THREADS
        self.is_running = False
        self.input_thread.join()
        self.output_thread.join()

    def test_video(self, path):
        if not os.path.exists(path):
            self.log("Video file does not exists!")
            return

        video = cv2.VideoCapture(path)
        video_length = video.get(cv2.CAP_PROP_FRAME_COUNT)
        i = 0
        fps = video.get(cv2.CAP_PROP_FPS)
        ret, img = video.read()
        while ret:
            start = time.time()
            key = cv2.waitKey(1)
            if key > -1:
                break
            self.log("{:.2f}%".format((i / (video_length - 1)) * 100))
            _, _, _ = self.get_frame(img)

            n_skip = int((time.time() - start) * fps)
            for _ in range(n_skip):
                _, _ = video.read()
                i += 1

            ret, img = video.read()
            i += 1
        self.log("100%")
        video.release()

    def forget_command(self, flag):
        self.ar.remove(flag)

    def learn_command(self, flag):
        # If a string is provided
        requires_focus = "-focus" in flag
        if '.' not in flag[0]:
            flag = flag[0]

            self.log("WAIT...")
            now = time.time()
            while (time.time() - now) < 3:
                _, _, _ = self.get_frame()

            self.log("GO!")
            playsound('assets' + os.sep + 'start.wav')
            poses = []
            i = 0
            while len(poses) < self.window_size:
                _, pose, _ = self.get_frame()
                if pose is not None:
                    poses.append(pose)
                self.log("{:.2f}%".format((i / (self.window_size - 1)) * 100))
                i += 1
            data = np.stack(poses)
            playsound('assets' + os.sep + 'stop.wav')
            self.log("100%")
        # If a path to a video is provided
        else:
            if not os.path.exists(flag[0]):
                self.log("Video file does not exist!")
                return
            # self.cap.release()
            video = cv2.VideoCapture(flag[0])
            poses = []
            fps = video.get(cv2.CAP_PROP_FPS)
            video_length = video.get(cv2.CAP_PROP_FRAME_COUNT)
            ret, img = video.read()
            i = 0
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
                    i += 1

                ret, img = video.read()
                self.log("{:.2f}%".format((i / (video_length - 1)) * 100))
                i += 1
            self.log("100%")
            video.release()
            flag = flag[0].split('/')[-1].split('.')[0]  # between / and .
            data = np.stack(poses)
            data = data[:(len(data) - (len(data) % self.window_size))]
            data = data[list(range(0, len(data), int(len(data) / self.window_size)))]

        self.log("Success!")
        data = (data, flag, requires_focus)
        self.ar.train(data)


if __name__ == "__main__":
    f = FocusDetector(FocusConfig())
    h = HumanPoseEstimator(MetrabsTRTConfig(), RealSenseIntrinsics())
    n = ActionRecognizer(TRXConfig())

    master = ISBFSAR(h, n, f, MainConfig(), debug=False)

    master.run()
