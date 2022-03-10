import threading
from multiprocessing.connection import Client
# import open3d as o3d
import numpy as np
import pyrealsense2 as rs


class JustText(threading.Thread):
    def __init__(self, port):
        # Connect to master
        super().__init__()
        self.conn = None
        self.port = port

    def run(self):
        address = ('localhost', self.port)
        self.conn = Client(address, authkey=b'secret password')
        while True:
            msg = input("Enter command here: ")
            if msg == "close" or msg == "exit" or msg == "quit":
                break
            self.conn.send(msg)


class RealSense:
    def __init__(self, width=640, height=480):
        self.pipeline = rs.pipeline()
        configs = {'device': 'Intel RealSense D435i'}

        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.profile = self.pipeline.start(config)

        configs['depth'] = {'width': width, 'height': height, 'format': 'z16', 'fps': 30}
        configs['color'] = {'width': width, 'height': height, 'format': 'rgb8', 'fps': 30}

        HIGH_ACCURACY = 3
        HIGH_DENSITY = 4
        MEDIUM_DENSITY = 5
        self.profile.get_device().sensors[0].set_option(rs.option.visual_preset, HIGH_DENSITY)

        configs['options'] = {}
        for device in self.profile.get_device().sensors:
            configs['options'][device.name] = {}
            for option in device.get_supported_options():
                configs['options'][device.name][str(option)[7:]] = str(device.get_option(option))

        self.configs = configs
        self.align = rs.align(rs.stream.depth)

    def intrinsics(self):
        return self.profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()

    def read(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)

        # depth_frame = aligned_frames.get_depth_frame()  # aligned_depth_frame is a 640x480 depth image
        # color_frame = aligned_frames.get_color_frame()
        depth_frame = frames.get_depth_frame()  # aligned_depth_frame is a 640x480 depth image
        color_frame = frames.get_color_frame()

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        return True, color_image
        # return color_image, depth_image

    @classmethod
    def pointcloud(cls, depth_image, rgb_image=None):
        if rgb_image is None:
            return cls._pointcloud(depth_image)

        depth_image = o3d.geometry.Image(depth_image)
        rgb_image = o3d.geometry.Image(rgb_image)
        rgbd = o3d.geometry.RGBDImage().create_from_color_and_depth(rgb_image, depth_image,
                                                                    convert_rgb_to_intensity=False,
                                                                    depth_scale=1000)

        intrinsics = {'fx': 384.025146484375, 'fy': 384.025146484375, 'ppx': 319.09661865234375,
                      'ppy': 237.75723266601562,
                      'width': 640, 'height': 480}

        camera = o3d.camera.PinholeCameraIntrinsic(intrinsics['width'], intrinsics['height'], intrinsics['fx'],
                                                   intrinsics['fy'], intrinsics['ppx'], intrinsics['ppy'])

        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, camera)
        pcd.transform([[1, 0, 0, 0],
                       [0, -1, 0, 0],
                       [0, 0, -1, 0],
                       [0, 0, 0, 1]])

        return pcd

    @classmethod
    def _pointcloud(cls, depth_image):
        depth_image = o3d.geometry.Image(depth_image)

        intrinsics = {'fx': 384.025146484375, 'fy': 384.025146484375, 'ppx': 319.09661865234375, 'ppy': 237.75723266601562,
                      'width': 640, 'height': 480}
        camera = o3d.camera.PinholeCameraIntrinsic(intrinsics['width'], intrinsics['height'], intrinsics['fx'],
                                                   intrinsics['fy'], intrinsics['ppx'], intrinsics['ppy'])

        pcd = o3d.geometry.PointCloud.create_from_depth_image(depth_image, camera)
        pcd.transform([[1, 0, 0, 0],
                       [0, -1, 0, 0],
                       [0, 0, -1, 0],
                       [0, 0, 0, 1]])

        return np.array(pcd.points)

    def stop(self):
        self.pipeline.stop()
