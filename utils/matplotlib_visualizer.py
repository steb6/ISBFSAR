import matplotlib.pyplot as plt
import numpy as np


class MPLPosePrinter:
    def __init__(self, **args):
        plt.ion()
        fig = plt.figure()
        self.ax = fig.add_subplot(111, projection='3d')

        # Setting the axes properties
        self.ax.set_xlim3d([-1.0, 1.0])
        self.ax.set_xlabel('X')

        self.ax.set_ylim3d([-1.0, 1.0])
        self.ax.set_ylabel('Y')

        self.ax.set_zlim3d([-1.0, 1.0])
        self.ax.set_zlabel('Z')

        self.ax.set_title('3D Test')
        # X = np.random.rand(30, 3)
        # self.sc = self.ax.scatter(X[:, 0], X[:, 1], X[:, 2])

        fig.show()

    def print_pose(self, pose, edges, color='b'):
        if len(pose.shape) == 2:
            pose = pose[None]
        # pose_flat = pose.reshape(-1, 3)
        # self.sc._offsets3d = (pose_flat[:, 0], pose_flat[:, 1], pose_flat[:, 2])
        if edges is not None:
            for p in pose:
                for edge in edges:
                    a = p[edge[0]]
                    b = p[edge[1]]
                    self.ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]], color)
        plt.draw()

    def clear(self):
        self.ax.lines.clear()

    @staticmethod
    def sleep(t):
        plt.pause(t)
