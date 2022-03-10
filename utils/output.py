import cv2
import numpy as np
from matplotlib import pyplot as plt
import open3d
from open3d.cpu.pybind.geometry import PointCloud, LineSet
from open3d.cpu.pybind.utility import Vector3dVector, Vector2iVector
from open3d.cpu.pybind.visualization import Visualizer
import open3d as o3d


class PosePrinter:
    def __init__(self, width, height, just_pose=False):
        # To visualize the pose
        self.vis = Visualizer()
        self.vis.create_window(width=width, height=height)
        self.render = True
        self.joints = PointCloud()
        self.lines = LineSet()
        # Just for debug
        self.coord = open3d.cpu.pybind.geometry.TriangleMesh.create_coordinate_frame()
        sphere_points = np.random.randn(3, 500)
        sphere_points /= np.linalg.norm(sphere_points, axis=0)
        sphere_points = sphere_points.swapaxes(0, 1)
        self.sphere = PointCloud()
        self.sphere.points = Vector3dVector(sphere_points)
        self.fps_s = []

        # Create room
        self.cam_height = 1060

        room_width = 2300
        room_height = 2850
        room_length = 4220

        x = room_width / 2

        room_xyz = np.array([[-x, 0, 0],
                             [x, 0, 0],
                             [x, room_height, 0],
                             [-x, room_height, 0],
                             [-x, 0, room_length],
                             [x, 0, room_length],
                             [x, room_height, room_length],
                             [-x, room_height, room_length]])
        room_connections = np.array([[0, 1],
                                     [1, 2],
                                     [2, 3],
                                     [3, 0],

                                     [4, 5],
                                     [5, 6],
                                     [6, 7],
                                     [7, 4],

                                     [6, 2],
                                     [7, 3],
                                     [5, 1],
                                     [4, 0]])
        room = LineSet()
        room.points = Vector3dVector(room_xyz)
        room.lines = Vector2iVector(room_connections)
        if not just_pose:
            self.vis.add_geometry(room)
            self.vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=100))
            self.vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=100,
                                                                                    origin=(0, self.cam_height, 0)))

    def print_pose(self, pose, edges):
        if pose is None:
            self.vis.poll_events()
            self.vis.update_renderer()
            return  # TODO change this line with 'pose = []' to change behaviour

        self.lines.points = Vector3dVector(pose)
        self.lines.lines = Vector2iVector(edges)

        aux = PointCloud()
        aux.points = Vector3dVector(pose)
        self.joints.clear()
        self.joints += aux

        if self.render:
            self.vis.add_geometry(self.joints)
            self.vis.add_geometry(self.lines)
            # vis.add_geometry(coord)
            # self.vis.add_geometry(self.sphere)
            self.render = False

        self.vis.update_geometry(self.joints)
        self.vis.update_geometry(self.lines)
        # vis.update_geometry(coord)
        # self.vis.update_geometry(self.sphere)
        self.vis.poll_events()
        self.vis.update_renderer()


def line_between(xs, ys, a, b, style):
    plt.plot((xs[a], xs[b]), (ys[a], ys[b]), style)


def print_points(points):  # 1 x 17 x 2, coordinates inverted

    plt.figure(1)
    plt.clf()

    plt.scatter(points[..., 1], points[..., 0])
    pose_x = points[:, 1]
    pose_y = points[:, 0]
    # face
    line_between(pose_x, pose_y, 0, 1, 'b')
    line_between(pose_x, pose_y, 0, 2, 'b')
    line_between(pose_x, pose_y, 1, 3, 'b')
    line_between(pose_x, pose_y, 2, 4, 'b')
    # legs
    line_between(pose_x, pose_y, 14, 16, 'b')
    line_between(pose_x, pose_y, 13, 15, 'b')
    line_between(pose_x, pose_y, 14, 12, 'b')
    line_between(pose_x, pose_y, 13, 11, 'b')
    # body
    line_between(pose_x, pose_y, 11, 5, 'b')
    line_between(pose_x, pose_y, 5, 6, 'b')
    line_between(pose_x, pose_y, 11, 12, 'b')
    line_between(pose_x, pose_y, 12, 6, 'b')
    # arms
    line_between(pose_x, pose_y, 7, 9, 'b')
    line_between(pose_x, pose_y, 7, 5, 'b')
    line_between(pose_x, pose_y, 6, 8, 'b')
    line_between(pose_x, pose_y, 8, 10, 'b')

    # Scores
    plt.draw()
    plt.pause(0.001)


def print_frame(f, bbox, h, w):
    x1 = int(bbox[0] * w)
    y1 = int(bbox[1] * h)
    x2 = int(bbox[2] * w)
    y2 = int(bbox[3] * h)
    # confidence = elem[4]
    # class_id = elem[5]
    f = cv2.rectangle(f, (x1, y1), (x2, y2), (255, 0, 0), 1)
    return f
