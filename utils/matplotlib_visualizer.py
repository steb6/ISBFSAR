import matplotlib.pyplot as plt
import numpy as np


class MPLPosePrinter:
    def __init__(self, **args):
        plt.ion()
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.view_init(elev=180, azim=0, vertical_axis='y')

        # Setting the axes properties
        self.ax.set_xlim3d([-1.0, 1.0])
        # self.ax.set_xlabel('X')

        self.ax.set_ylim3d([-1.0, 1.0])
        # self.ax.set_ylabel('Y')

        self.ax.set_zlim3d([-1.0, 1.0])
        # self.ax.set_zlabel('Z')

        # self.ax.set_title('3D Test')
        plt.grid(False)
        plt.axis('off')
        self.fig.show()

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

    def set_title(self, title):
        self.ax.set_title(title)

    def save(self, path):
        import cv2
        path = path + ".png"
        self.fig.savefig(path)
        img = cv2.imread(path)
        # cv2.imshow("", img)
        # cv2.waitKey(0)
        img = img[200:280, 280:360, :]
        cv2.imwrite(path, img)
