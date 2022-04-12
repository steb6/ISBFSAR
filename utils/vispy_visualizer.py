import multiprocessing
from queue import Empty
import numpy as np
from vispy import app, scene, visuals
from threading import Thread
import time


class VISPYVisualizer:

    def __init__(self, queue):

        self.queue = queue
        self._timer = app.Timer('auto', connect=self.on_timer, start=True)

        canvas = scene.SceneCanvas(keys="interactive", title="plot3d", show=True)
        view = canvas.central_widget.add_view()
        view.camera = "turntable"
        view.camera.fov = 45
        view.camera.distance = 3
        self.lines = []
        Plot3D = scene.visuals.create_visual_node(visuals.LinePlotVisual)
        for _ in range(30):
            self.lines.append(Plot3D(
                [],
                width=2.0,
                color="red",
                edge_color="w",
                symbol="o",
                face_color=(0.2, 0.2, 1, 0.8),
                parent=view.scene,
            ))

    def on_timer(self, _):
        try:
            points, edges, image = self.queue.get_nowait()
        except Empty:
            return
        for i, edge in enumerate(edges):
            self.lines[i].set_data((points[[edge[0], edge[1]]]))

    def on_draw(self, event):
        pass


if __name__ == '__main__':

    def create_visualizer(qe):
        canvas = VISPYVisualizer(qe)
        app.run()


    q = multiprocessing.Queue()
    Thread(target=create_visualizer, args=(q,)).start()

    N = 60
    x = np.sin(np.linspace(-2, 2, N) * np.pi)
    y = np.cos(np.linspace(-2, 2, N) * np.pi)
    z = np.linspace(-2, 2, N)
    pos = np.c_[x, y, z]

    for _ in range(1000):
        print("MIAO")
        q.put(pos[:_])
        time.sleep(1)
