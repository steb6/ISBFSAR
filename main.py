import pickle as pkl
from modules.focus.gaze_estimation.focus import FocusDetector
# from modules.focus.mutual_gaze.focus import FocusDetector
import os
import numpy as np
from tqdm import tqdm
import time
from modules.ar.ar import ActionRecognizer
import cv2
from playsound import playsound
from utils.input import RealSense, just_text
from modules.hpe.hpe import HumanPoseEstimator
from utils.params import MetrabsTRTConfig, RealSenseIntrinsics, MainConfig, FocusConfig
from utils.params import TRXConfig
from utils.output import VISPYVisualizer
from multiprocessing import Process, Queue
from threading import Thread


class ISBFSAR:
    def __init__(self, args, visualizer=True, video_input=None):
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
        # TODO IF I USE PROCESS IT WORKS BUT EXCEPT, WITH THREADS IT DOESNT WORK
        self.input_queue = Queue(1)
        # self.input_proc = Process(target=just_text, args=(self.input_queue,))
        # self.input_proc.start()

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

        self.hpe_in.put(img)
        self.focus_in.put(img)

        pose3d_abs, edges, bbox = self.hpe_out.get()
        focus_ret = self.focus_out.get()

        if focus_ret is not None:
            focus, face = focus_ret
            # img = self.focus.print_bbox(img, face)  # TODO PRINT FACE AGAIN

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
        actions, is_true = results

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

        return img, pose3d_root, results

    def log(self, msg):
        self.output_queue.put(({"log": msg},))
        print(msg)

    def run(self):
        while True:
        # for _ in tqdm(list(range(10000))):
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

            elif msg[0] == "sim":
                self.sim(msg[1], msg[2])

            elif msg[0] == "test_dir":  # TODO REMOVE DEBUG
                self.test_dir(msg[1])  # TODO REMOVE DEBUG

            else:
                self.log("Not a valid command!")

        # clean  # TODO TERMINATE THREADS
        self.input_proc.join()
        self.output_proc.join()

    # TODO REMOVE DEBUG
    def test_dir(self, path):
        poses = []
        for i in range(16):
            img = cv2.imread(os.path.join(path, "img_{}.png".format(i)))
            _, pose, _ = self.get_frame(img)
            poses.append(pose)

        poses = np.stack(poses)
        for pose in poses:
            results = self.ar.inference(pose)
            actions, is_true = results
            print(actions)
            print(is_true)
    # TODO END REMOVE DEBUG

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
        requires_box = None
        if "-box" in flag:
            requires_box = True
        if "-nobox" in flag:
            requires_box = False
        if '.' not in flag[0]:
            flag = flag[0]

            self.log("WAIT...")
            now = time.time()
            while (time.time() - now) < 3:
                _, _, _ = self.get_frame()

            self.log("GO!")
            playsound('assets' + os.sep + 'start.wav')
            poses = []
            imgs = []  # TODO REMOVE DEBUG
            i = 0
            while len(poses) < self.window_size:
                img, pose, _ = self.get_frame()
                imgs.append(img)  # TODO REMOVE DEBUG
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

        # TODO SAVE POSE AND
        import pickle
        out_dir = "testing"
        skeleton = 'smpl+head_30'
        with open('assets/skeleton_types.pkl', "rb") as input_file:
            skeleton_types = pickle.load(input_file)
        edges = skeleton_types[skeleton]['edges']
        from utils.matplotlib_visualizer import MPLPosePrinter
        vis = MPLPosePrinter()
        os.mkdir(os.path.join("imgs", flag))
        for i, img in enumerate(imgs):
            img = img[::-1, :, ::-1]
            cv2.imwrite(os.path.join("imgs", flag, f"img_{i}.png"), img)
        for j, pose in enumerate(poses):
            vis.clear()
            vis.print_pose(pose.reshape(-1, 3), edges)
            vis.save(os.path.join("imgs", flag, f"pose_{j}.png"))
            vis.sleep(0.01)
        # TODO END
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
        with open('assets/saved/requires_box.pkl', 'wb') as outfile:
            pkl.dump(self.ar.requires_box, outfile)
        with open('assets/saved/sim.pkl', 'wb') as outfile:
            pkl.dump(self.ar.similar_actions, outfile)

    def load(self):
        with open('assets/saved/support_set.pkl', 'rb') as pkl_file:
            self.ar.support_set = pkl.load(pkl_file)
        with open('assets/saved/support_labels.pkl', 'rb') as pkl_file:
            self.ar.support_labels = pkl.load(pkl_file)
        with open('assets/saved/requires_focus.pkl', 'rb') as pkl_file:
            self.ar.requires_focus = pkl.load(pkl_file)
        with open('assets/saved/requires_box.pkl', 'rb') as pkl_file:
            self.ar.requires_box = pkl.load(pkl_file)
        with open('assets/saved/sim.pkl', 'rb') as pkl_file:
            self.ar.similar_actions = pkl.load(pkl_file)

    def sim(self, action1, action2):
        self.ar.sim(action1, action2)


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
