from vispy import app, scene, visuals
from vispy.scene.visuals import Text, Image
import numpy as np
import math


class VISPYVisualizer:

    def printer(self, x):
        if x.text == '\b':
            self.input_text = self.input_text[:-1]
            self.log.text = ''
        elif x.text == '\r':
            self.output_queue.put(self.input_text[1:])  # Do not send '<'
            self.input_text = '>'
            self.log.text = ''
        elif x.text == '\\':
            self.show = not self.show
        else:
            self.input_text += x.text
        self.input_string.text = self.input_text

    def __init__(self, input_queue, output_queue):

        self.input_queue = input_queue
        self.output_queue = output_queue
        self.show = True

        self._timer = app.Timer('auto', connect=self.on_timer, start=True)
        self.input_text = '>'

        canvas = scene.SceneCanvas(keys='interactive')
        canvas.size = 1200, 600
        canvas.show()
        canvas.events.key_press.connect(self.printer)

        # This is the top-level widget that will hold three ViewBoxes, which will
        # be automatically resized whenever the grid is resized.
        grid = canvas.central_widget.add_grid()

        # Plot
        b1 = grid.add_view(row=0, col=0)
        b1.border_color = (0.5, 0.5, 0.5, 1)
        # b1.camera = scene.TurntableCamera(45, elevation=-90, azimuth=0, distance=2)  # TODO OLD
        b1.camera = scene.TurntableCamera(45, elevation=30, azimuth=0, distance=2)
        self.lines = []
        Plot3D = scene.visuals.create_visual_node(visuals.LinePlotVisual)
        for _ in range(30):
            self.lines.append(Plot3D(
                [],
                width=3.0,
                color="purple",
                edge_color="w",
                symbol="o",
                face_color=(0.2, 0.2, 1, 0.8),
                marker_size=1,
            ))
            b1.add(self.lines[_])
        coords = scene.visuals.GridLines(parent=b1.scene)

        # Info
        self.b2 = grid.add_view(row=0, col=1)
        self.b2.camera = scene.PanZoomCamera(rect=(0, 0, 1, 1))
        self.b2.camera.interactive = False
        self.b2.border_color = (0.5, 0.5, 0.5, 1)
        self.fps = Text('FPS:', color='white', rotation=0, anchor_x="center", anchor_y="bottom",
                        font_size=12, pos=(0.75, 0.9))
        self.focus = Text('Non focus', color='green', rotation=0, anchor_x="center", anchor_y="bottom",
                          font_size=12, pos=(0.25, 0.9))
        self.b2.add(self.fps)
        self.b2.add(self.focus)
        self.actions = {}

        # Image
        b3 = grid.add_view(row=1, col=0)
        b3.camera = scene.PanZoomCamera(rect=(0, 0, 640, 480))
        b3.camera.interactive = False
        b3.border_color = (0.5, 0.5, 0.5, 1)
        self.image = Image()
        b3.add(self.image)

        # Commands
        b4 = grid.add_view(row=1, col=1)
        b4.camera = scene.PanZoomCamera(rect=(0, 0, 1, 1))
        b4.camera.interactive = False
        b4.border_color = (0.5, 0.5, 0.5, 1)
        self.desc_add = Text('ADD ACTION: add action_name [-focus]', color='white', rotation=0, anchor_x="left",
                             anchor_y="bottom",
                             font_size=12, pos=(0.1, 0.9))
        self.desc_remove = Text('REMOVE ACTION: remove action_name', color='white', rotation=0, anchor_x="left",
                                anchor_y="bottom",
                                font_size=12, pos=(0.1, 0.7))
        self.input_string = Text(self.input_text, color='purple', rotation=0, anchor_x="left", anchor_y="bottom",
                                 font_size=12, pos=(0.1, 0.5))
        self.log = Text('', color='orange', rotation=0, anchor_x="left", anchor_y="bottom",
                        font_size=12, pos=(0.1, 0.3))
        b4.add(self.desc_add)
        b4.add(self.desc_remove)
        b4.add(self.input_string)
        b4.add(self.log)

    def on_timer(self, _):
        # Check if visualized is disabled
        if not self.show:
            return
        # Check if there is something to show
        elements = []
        while not self.input_queue.empty():
            elements.append(self.input_queue.get())
        if len(elements) == 0:
            return
        elements = elements[-1]
        # Parse elements
        elements = elements[0]
        if "log" in elements.keys():
            self.log.text = elements["log"]
        else:
            edges = elements["edges"]
            pose = elements["pose"]
            img = elements["img"]
            focus = elements["focus"]
            fps_without_vis = elements["fps_without_vis"]
            results = elements["actions"]
            # Rotate pose of 90 degree along y axe
            theta = 90
            R = np.matrix([[1, 0, 0],
                           [0, math.cos(theta), -math.sin(theta)],
                           [0, math.sin(theta), math.cos(theta)]])
            pose = pose @ R
            # pose[:, 1] += min(pose[:, 1])
            for i, edge in enumerate(edges):
                self.lines[i].set_data((pose[[edge[0], edge[1]]]))
            self.image.set_data(img)
            if focus:
                self.focus.text = "FOCUS"
                self.focus.color = "green"
            else:
                self.focus.text = "NOT FOCUS"
                self.focus.color = "red"
            self.fps.text = "FPS: {:.2f}".format(fps_without_vis)
            # Print action
            i = 0
            m = max([_[0] for _ in results.values()]) if len(results) > 0 else 0
            for r in results.keys():
                score, requires_focus = results[r]
                if score == m:
                    if requires_focus:
                        if focus:
                            color = "green"
                        else:
                            color = "orange"
                    else:
                        color = "green"
                else:
                    color = "red"
                if r in self.actions.keys():
                    self.actions[r].text = "{}: {:.2f}, {}".format(r, score, requires_focus)
                else:
                    self.actions[r] = Text('', rotation=0, anchor_x="center", anchor_y="bottom", font_size=12)
                    self.b2.add(self.actions[r])
                self.actions[r].pos = 0.5, 0.7 - (0.1 * i)
                self.actions[r].color = color
                i += 1
            # Remove erased action (if any)
            to_remove = []
            for key in self.actions.keys():
                if key not in results.keys():
                    to_remove.append(key)
            for key in to_remove:
                self.actions[key].parent = None
                self.actions.pop(key)

    def on_draw(self, event):
        pass
