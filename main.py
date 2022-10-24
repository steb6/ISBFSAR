import pickle as pkl
from multiprocessing.managers import BaseManager
from modules.focus.gaze_estimation.focus import FocusDetector
# from modules.focus.mutual_gaze.focus import FocusDetector
import os
import numpy as np
import time
from modules.ar.ar import ActionRecognizer
import cv2
from playsound import playsound
from modules.hpe.hpe import HumanPoseEstimator
from utils.params import MetrabsTRTConfig, RealSenseIntrinsics, MainConfig, FocusConfig
from utils.params import TRXConfig
from multiprocessing import Process, Queue


docker = os.environ.get('AM_I_IN_A_DOCKER_CONTAINER', False)


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

        self.ar = ActionRecognizer(TRXConfig(), add_hook=False)

        # Create communication with host
        BaseManager.register('get_queue')
        manager = BaseManager(address=("host.docker.internal" if docker else "localhost", 50000), authkey=b'abracadabra')
        manager.connect()
        self._in_queue = manager.get_queue('src_to_sink')  # To get rgb or msg
        self._out_queue = manager.get_queue('sink_to_src')  # To send element to VISPY

        # Variables
        self.cam_width = args.cam_width
        self.cam_height = args.cam_height
        self.window_size = args.window_size
        self.fps_s = []
        self.last_poses = []
        self.skeleton_scale = args.skeleton_scale
        self.acquisition_time = args.acquisition_time
        self.edges = None

    def get_frame(self, img=None, log=None):
        """
        get frame, do inference, return all possible info
        """
        start = time.time()
        elements = {}
        ar_input = {}

        # If img is not given (not a video), try to get img
        if img is None:
            img = self._in_queue.get()["rgb"]
        elements["img"] = img

        # Start independent modules
        self.focus_in.put(img)
        self.hpe_in.put(img)

        # RGB CASE
        hpe_res = self.hpe_out.get()
        if self.input_type == "hybrid" or self.input_type == "rgb":
            if hpe_res is not None:
                x1, x2, y1, y2 = hpe_res['bbox']
                elements["bbox"] = x1, x2, y1, y2
                xm = int((x1 + x2) / 2)
                ym = int((y1 + y2) / 2)
                l = max(xm - x1, ym - y1)
                img_ = img[(ym - l if ym - l > 0 else 0):(ym + l), (xm - l if xm - l > 0 else 0):(xm + l)]
                img_ = cv2.resize(img_, (224, 224))
                # cv2.imshow("", img_)  # TODO REMOVE DEBUG
                # cv2.waitKey(1)  # TODO REMOVE DEBUG
                img_ = img_ / 255.
                img_ = img_ * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                img_ = img_.swapaxes(-1, -3).swapaxes(-1, -2)
                ar_input["rgb"] = img_
                elements["img_preprocessed"] = img_

        # SKELETON CASE
        if self.input_type == "hybrid" or self.input_type == "skeleton":
            if hpe_res is not None:
                pose, edges, bbox = hpe_res['pose'], hpe_res['edges'], hpe_res['bbox']
                if self.edges is None:
                    self.edges = edges
                if pose is not None:
                    elements["distance"] = np.sqrt(np.sum(np.square(np.array([0, 0, 0]) - np.array(pose[0])))) * 2.5
                    pose = pose - pose[0, :]
                    elements["pose"] = pose
                    ar_input["sk"] = pose.reshape(-1)
                elements["edges"] = edges
                if bbox is not None:
                    elements["bbox"] = bbox

        # Make inference
        results = self.ar.inference(ar_input)
        actions, is_true, requires_focus = results
        elements["actions"] = actions
        elements["is_true"] = is_true
        elements["requires_focus"] = requires_focus

        # FOCUS #######################################################
        focus_ret = self.focus_out.get()
        if focus_ret is not None:
            focus, face = focus_ret
            elements["focus"] = focus
            elements["face_bbox"] = face.bbox.reshape(-1)

        end = time.time()

        # Compute fps
        self.fps_s.append(1. / (end - start))
        fps_s = self.fps_s[-10:]
        fps = sum(fps_s) / len(fps_s)
        elements["fps"] = fps

        # Msg
        if log is not None:
            elements["log"] = log

        self._out_queue.put(elements)

        return elements

    def run(self):
        while True:
            log = None
            data = self._in_queue.get()

            if data["msg"] != '':

                msg = data["msg"]
                msg = msg.strip()
                msg = msg.split()

                # select appropriate command
                if msg[0] == 'close' or msg[0] == 'exit' or msg[0] == 'quit' or msg[0] == 'q':
                    break

                elif msg[0] == "add" and len(msg) > 1:
                    self._out_queue.put({"ACK": True})  # This must be sent as answer to not stop the program
                    log = self.learn_command(msg[1:])
                    data = self._in_queue.get()

                elif msg[0] == "remove" and len(msg) > 1:
                    log = self.forget_command(msg[1])

                # elif msg[0] == "test" and len(msg) > 1:
                #     self.test_video(msg[1])

                elif msg[0] == "save":
                    log = self.save()

                elif msg[0] == "load":
                    log = self.load()

                elif msg[0] == "debug":
                    self.debug()
                else:
                    log = "Not a valid command!"

            self.get_frame(img=data["rgb"], log=log)

    # def test_video(self, path):
    #     if not os.path.exists(path):
    #         self.log("Video file does not exists!")
    #         return
    #
    #     video = cv2.VideoCapture(path)
    #     video_length = video.get(cv2.CAP_PROP_FRAME_COUNT)
    #     i = 0
    #     fps = video.get(cv2.CAP_PROP_FPS)
    #     ret, img = video.read()
    #     while ret:
    #         start = time.time()
    #         key = cv2.waitKey(1)
    #         if key > -1:
    #             break
    #         self.log("{:.2f}%".format((i / (video_length - 1)) * 100))
    #         _ = self.get_frame(img)
    #
    #         n_skip = int((time.time() - start) * fps)
    #         for _ in range(n_skip):
    #             _, _ = video.read()
    #             i += 1
    #
    #         ret, img = video.read()
    #         i += 1
    #     self.log("100%")
    #     video.release()

    def forget_command(self, flag):
        if self.ar.remove(flag):
            return "Action {} removed".format(flag)
        else:
            return "Action {} is not in the support set".format(flag)

    def debug(self):
        ss = self.ar.support_set
        if len(ss) == 0:
            return
        if self.input_type in ["hybrid", "imgs"]:
            ss = np.stack([ss[c]["imgs"].detach().cpu().numpy() for c in ss.keys()])
            ss = ss.swapaxes(-2, -3).swapaxes(-1, -2)
            ss = (ss - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
            ss = (ss * 255).astype(np.uint8)
            n = len(ss)
            cv2.imshow("support_set_RGB",
                       cv2.resize(ss.swapaxes(0, 1).reshape(8, 224 * n, 224, 3).swapaxes(0, 1).reshape(n * 224, 8 * 224, 3),
                                  (640, 96 * len(ss))))
        if self.input_type in ["hybrid", "skeleton"]:
            ss = np.stack([ss[c]["poses"].detach().cpu().numpy() for c in ss.keys()])
            ss = ss.reshape(ss.shape[:-1]+(30, 3))
            size = 100
            visual = np.zeros((size*ss.shape[0], size*ss.shape[1]))
            ss = (ss + 1)*(size/2)  # Send each pose from [-1, +1] to [0, size]
            ss = ss[..., :2]
            ss[..., 1] += np.arange(ss.shape[0])[..., None, None].repeat(ss.shape[1], axis=1)*size
            ss[..., 0] += np.arange(ss.shape[1])[None, ..., None].repeat(ss.shape[0], axis=0)*size
            ss = ss.reshape(-1, 30, 2).astype(int)
            for pose in ss:
                for point in pose:
                    visual = cv2.circle(visual, point, 1, (255, 0, 0))
                for edge in self.edges:
                    visual = cv2.line(visual, pose[edge[0]], pose[edge[1]], (255, 0, 0))
            cv2.imshow("support_set_SK", visual)
        cv2.waitKey(0)

    def learn_command(self, flag):
        requires_focus = "-focus" in flag
        flag = flag[0]
        now = time.time()
        while (time.time() - now) < 3:
            self.get_frame(log="WAIT...")

        self.get_frame(log="GO!")
        # playsound('assets' + os.sep + 'start.wav')
        data = [[] for _ in range(self.window_size)]
        i = 0
        off_time = (self.acquisition_time / self.window_size)
        while i < self.window_size:
            start = time.time()
            res = self.get_frame(log="{:.2f}%".format((i / (self.window_size - 1)) * 100))
            # Check if the sample is good w.r.t. input type
            good = self.input_type in ["skeleton", "hybrid"] and "pose" in res.keys() and res["pose"] is not None
            good = good or self.input_type == "rgb"
            if good:
                if self.input_type in ["skeleton", "hybrid"]:
                    data[i].append(res["pose"].reshape(-1))  # CAREFUL with the reshape
                if self.input_type in ["rgb", "hybrid"]:
                    data[i].append(res["img_preprocessed"])
                i += 1
            while (time.time() - start) < off_time:  # Busy wait
                continue

        # playsound('assets' + os.sep + 'stop.wav')
        # self.log("100%")

        # If a path to a video is provided
        # else:
        #     if not os.path.exists(flag[0]):
        #         self.log("Video file does not exist!")
        #         return
        #     # self.cap.release()
        #     video = cv2.VideoCapture(flag[0])
        #     poses = []
        #     fps = video.get(cv2.CAP_PROP_FPS)
        #     video_length = video.get(cv2.CAP_PROP_FRAME_COUNT)
        #     ret, img = video.read()
        #     i = 0
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
        #             i += 1
        #
        #         ret, img = video.read()
        #         self.log("{:.2f}%".format((i / (video_length - 1)) * 100))
        #         i += 1
        #     self.log("100%")
        #     video.release()
        #     flag = flag[0].split('/')[-1].split('.')[0]  # between / and .
        #     data = np.stack(poses)
        #     data = data[:(len(data) - (len(data) % self.window_size))]
        #     data = data[list(range(0, len(data), int(len(data) / self.window_size)))]
        inp = {"flag": flag,
               "data": {},
               "requires_focus": requires_focus}

        if self.input_type == "rgb":  # Unique case with images in first position
            inp["data"]["imgs"] = np.stack([x[0] for x in data])
        if self.input_type in ["skeleton", "hybrid"]:
            inp["data"]["poses"] = np.stack([x[0] for x in data])
        if self.input_type == "hybrid":
            inp["data"]["imgs"] = np.stack([x[1] for x in data])
        self.ar.train(inp)
        return "Action " + flag + " learned successfully!"

    def save(self):
        with open('assets/saved/support_set.pkl', 'wb') as outfile:
            pkl.dump(self.ar.support_set, outfile)
        with open('assets/saved/requires_focus.pkl', 'wb') as outfile:
            pkl.dump(self.ar.requires_focus, outfile)
        return "Classes saved successfully in " + 'assets/saved/support_set.pkl'

    def load(self):
        with open('assets/saved/support_set.pkl', 'rb') as pkl_file:
            self.ar.support_set = pkl.load(pkl_file)
        with open('assets/saved/requires_focus.pkl', 'rb') as pkl_file:
            self.ar.requires_focus = pkl.load(pkl_file)
        return f"Loaded {len(self.ar.support_set)} classes"


def run_module(module, configurations, input_queue, output_queue):
    import pycuda.autoinit
    x = module(*configurations)
    while True:
        inp = input_queue.get()
        y = x.estimate(inp)
        output_queue.put(y)


if __name__ == "__main__":
    master = ISBFSAR(MainConfig(), visualizer=True)
    master.run()
