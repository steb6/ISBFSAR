import pickle as pkl
from modules.focus.gaze_estimation.focus import FocusDetector
# from modules.focus.mutual_gaze.focus import FocusDetector
import os
import numpy as np
import time
from modules.ar.ar import ActionRecognizer
import cv2
from playsound import playsound
from utils.input import RealSense
from modules.hpe.hpe import HumanPoseEstimator
from utils.params import MetrabsTRTConfig, RealSenseIntrinsics, MainConfig, FocusConfig
from utils.params import TRXConfig
from utils.output import VISPYVisualizer
from multiprocessing import Process, Queue


class ISBFSAR:
    def __init__(self, args, visualizer=True, video_input=None):
        self.input_type = args.input_type

        # Load modules
        self.focus_in = Queue(1)
        self.focus_out = Queue(1)
        self.focus_proc = Process(target=run_module, args=(FocusDetector,
                                                           (FocusConfig(),),
                                                           self.focus_in, self.focus_out))
        self.focus_proc.start()

        self.hpe_in = Queue(1)
        self.hpe_out = Queue(1)
        self.hpe_proc = Process(target=run_module, args=(HumanPoseEstimator,
                                                         (MetrabsTRTConfig(), RealSenseIntrinsics()),
                                                         self.hpe_in, self.hpe_out))
        self.hpe_proc.start()

        self.ar = ActionRecognizer(TRXConfig())

        # Connect to webcam
        if video_input is None:
            if args.cam == "webcam":
                self.cap = cv2.VideoCapture(0)
                self.cap.set(3, args.cam_width)
                self.cap.set(4, args.cam_height)
            elif args.cam == "realsense":
                self.cap = RealSense(width=args.cam_width, height=args.cam_height, fps=60)
                # intrinsics = self.cap.intrinsics()
                # i = np.eye(3)
                # i[0][0] = intrinsics.fx
                # i[0][2] = intrinsics.ppx
                # i[1][1] = intrinsics.fy
                # i[1][2] = intrinsics.ppy
                # self.hpe.intrinsics = i
        else:
            self.cap = cv2.VideoCapture(video_input)

        self.cam_width = args.cam_width
        self.cam_height = args.cam_height
        self.window_size = args.window_size
        self.fps_s = []
        self.last_poses = []
        self.skeleton_scale = args.skeleton_scale

        # Create input
        self.input_queue = Queue(1)

        # Create output
        self.visualizer = visualizer
        if self.visualizer:
            self.output_queue = Queue(1)
            self.output_proc = Process(target=VISPYVisualizer.create_visualizer,
                                       args=(self.output_queue, self.input_queue))
            self.output_proc.start()

    def get_frame(self, img=None):
        start = time.time()

        # If img is not given (not a video), try to get img
        if img is None:
            ret, img = self.cap.read()
            if not ret:
                raise Exception("Cannot grab frame!")

        # Start independent modules
        focus = False

        self.focus_in.put(img)

        # AR ############################################################
        self.hpe_in.put(img)

        if self.input_type == "skeleton":
            h = self.hpe_out.get()
            if h is not None:
                pose3d_abs, edges, bbox = h
            else:
                pose3d_abs, edges, bbox = None, None, None

            # Compute distance
            d = None
            if pose3d_abs is not None:
                cam_pos = np.array([0, 0, 0])
                man_pose = np.array(pose3d_abs[0])
                d = np.sqrt(np.sum(np.square(cam_pos - man_pose)))

            # Normalize
            pose3d_root = pose3d_abs - pose3d_abs[0, :] if pose3d_abs is not None else None
            ar_input = pose3d_root

        else:  # RGB case
            h = self.hpe_out.get()
            if h is not None:
                x1, y1, x2, y2 = h
                xm = int((x1 + x2) / 2)
                ym = int((y1 + y2) / 2)
                l = max(xm - x1, ym - y1)
                ar_input = img[(ym - l if ym - l > 0 else 0):(ym + l), (xm - l if xm - l > 0 else 0):(xm + l)]
                ar_input = cv2.resize(ar_input, (224, 224))
                cv2.imshow("AR_INPUT", ar_input)  # TODO REMOVE DEBUG
                cv2.waitKey(1)  # TODO REMOVE DEBUG
                ar_input = ar_input / 255.
                ar_input = ar_input * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                ar_input = ar_input.swapaxes(-1, -3).swapaxes(-1, -2)
                bbox = x1, x2, y1, y2
            else:
                ar_input = None
                bbox = None
            pose3d_root = None
            edges = None
            d = None

        # Make inference
        results = self.ar.inference(ar_input)
        actions, is_true = results

        # FOCUS #######################################################
        focus_ret = self.focus_out.get()
        if focus_ret is not None:
            focus, face = focus_ret
            # img = self.focus.print_bbox(img, face)  # TODO PRINT FACE AGAIN

        end = time.time()

        # Compute fps
        self.fps_s.append(1. / (end - start))
        fps_s = self.fps_s[-10:]
        fps = sum(fps_s) / len(fps_s)

        if self.visualizer:
            if bbox is not None:
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
                        "actions": actions,
                        "is_true": is_true,
                        "distance": d * 2 if d is not None else None,
                        "box": None
                        }
            self.output_queue.put((elements,))

        return ar_input, pose3d_root, results

    def log(self, msg):
        self.output_queue.put(({"log": msg},))
        print(msg)

    def run(self):
        while True:
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

            elif msg[0] == "save":
                self.save()

            elif msg[0] == "load":
                self.load()

            elif msg[0] == "debug":
                self.debug()

            else:
                self.log("Not a valid command!")

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

    def debug(self):
        support_set = self.ar.support_set
        support_set = support_set.detach().cpu().numpy().swapaxes(-2, -3).swapaxes(-1, -2)
        support_set = (support_set - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        # support_set = support_set.swapaxes(-1, -2).swapaxes(-2, -3)
        support_set = (support_set * 255).astype(np.uint8)
        support_set = support_set.swapaxes(0, 1).reshape(8, 224*5, 224, 3).swapaxes(0, 1).reshape(5*224, 8*224, 3)
        cv2.imshow("support_set", support_set)
        cv2.waitKey(1)

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
            imgs = []
            i = 0
            off_time = 3000 / self.window_size
            while True:
                start = time.time()
                if len(poses if self.input_type == "skeleton" else imgs) == self.window_size:
                    break
                img, pose, _ = self.get_frame()
                imgs.append(img)
                if pose is not None:
                    poses.append(pose)
                self.log("{:.2f}%".format((i / (self.window_size - 1)) * 100))
                i += 1
                while (time.time() - start)*1000 < off_time:
                    continue

            data = np.stack(poses if self.input_type == "skeleton" else imgs)
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
        data = (data, flag)
        self.ar.train(data)

    def save(self):
        with open('assets/saved/support_set.pkl', 'wb') as outfile:
            pkl.dump(self.ar.support_set, outfile)
        with open('assets/saved/support_labels.pkl', 'wb') as outfile:
            pkl.dump(self.ar.support_labels, outfile)
        with open('assets/saved/requires_focus.pkl', 'wb') as outfile:
            pkl.dump(self.ar.requires_focus, outfile)

    def load(self):
        with open('assets/saved/support_set.pkl', 'rb') as pkl_file:
            self.ar.support_set = pkl.load(pkl_file)
        with open('assets/saved/support_labels.pkl', 'rb') as pkl_file:
            self.ar.support_labels = pkl.load(pkl_file)
        with open('assets/saved/requires_focus.pkl', 'rb') as pkl_file:
            self.ar.requires_focus = pkl.load(pkl_file)
        self.ar.n_classes = len(list(filter(lambda x: x is not None, self.ar.support_labels)))
        print("Loaded", self.ar.n_classes, "classes")


def run_module(module, configurations, input_queue, output_queue):
    x = module(*configurations)
    while True:
        inp = input_queue.get()
        y = x.estimate(inp)
        output_queue.put(y)


if __name__ == "__main__":
    master = ISBFSAR(MainConfig(), visualizer=True)
    # master = ISBFSAR(h, n, f, MainConfig(), visualizer=True, video_input="assets/test_gaze_no_mask.mp4")

    master.run()
