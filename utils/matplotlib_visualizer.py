import matplotlib


class MPLPosePrinter:
    def __init__(self):
        matplotlib.pyplot.ion()
        fig = matplotlib.pyplot.figure()
        matplotlib.pyplot.axis('off')

        # Add pose axe
        self.pose_ax = fig.add_subplot(121, projection='3d')
        self.pose_ax.view_init(90, 90)
        self.pose_ax.set_xlim3d([-1.0, 1.0])
        self.pose_ax.set_xlabel('')
        self.pose_ax.set_ylim3d([-1.0, 1.0])
        self.pose_ax.set_ylabel('')
        self.pose_ax.set_zlim3d([-1.0, 1.0])
        self.pose_ax.set_zlabel('')
        self.pose_ax.set_title('3D pose')
        self.pose_ax.set_xticks([], [])
        self.pose_ax.set_yticks([], [])
        self.pose_ax.set_zticks([], [])

        # Add image axe
        self.image_ax = fig.add_subplot(122)

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
                    self.pose_ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]], color)

    def print_image(self, img):
        self.image_ax.imshow(img)
        self.image_ax.set_xticks([], [])
        self.image_ax.set_yticks([], [])

    @staticmethod
    def draw():
        matplotlib.pyplot.draw()

    @staticmethod
    def sleep(t):
        matplotlib.pyplot.pause(t)

    def clear(self):
        self.pose_ax.lines.clear()
        self.image_ax.clear()

    def set_title(self, title):
        self.pose_ax.set_title(title)
