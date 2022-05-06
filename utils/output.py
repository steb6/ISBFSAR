from vispy import app, scene, visuals
from vispy.scene.visuals import Text, Image
import numpy as np
import math


class VISPYVisualizer:

    def printer(self, x):
        if x.text == '\b':
            if len(self.input_text) > 1:
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

    @staticmethod
    def create_visualizer(qi, qo):
        _ = VISPYVisualizer(qi, qo)
        app.run()

    def __init__(self, input_queue, output_queue):

        self.input_queue = input_queue
        self.output_queue = output_queue
        self.show = True

        self._timer = app.Timer('auto', connect=self.on_timer, start=True)
        self.input_text = '>'

        self.canvas = scene.SceneCanvas(keys='interactive')
        self.canvas.size = 1200, 600
        self.canvas.show()
        self.canvas.events.key_press.connect(self.printer)

        # This is the top-level widget that will hold three ViewBoxes, which will
        # be automatically resized whenever the grid is resized.
        grid = self.canvas.central_widget.add_grid()

        # Plot
        b1 = grid.add_view(row=0, col=0)
        b1.border_color = (0.5, 0.5, 0.5, 1)
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
        self.distance = Text('', color='white', rotation=0, anchor_x="center", anchor_y="bottom",
                             font_size=12, pos=(0.25, 0.9))
        self.b2.add(self.distance)
        self.focus = Text('', color='green', rotation=0, anchor_x="center", anchor_y="bottom",
                          font_size=12, pos=(0.5, 0.9))
        self.b2.add(self.focus)
        self.fps = Text('', color='white', rotation=0, anchor_x="center", anchor_y="bottom",
                        font_size=12, pos=(0.75, 0.9))
        self.b2.add(self.fps)
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
        self.desc_add = Text('ADD ACTION: add action_name [-focus][-box/nobox]', color='white', rotation=0, anchor_x="left",
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
        # if not self.is_running:
        #     self.canvas.close()
        #     exit()
        if not self.show:
            return
        # Check if there is something to show
        elements = self.input_queue.get()
        if not elements:
            return
        # Parse elements
        elements = elements[0]
        if "log" in elements.keys():
            self.log.text = elements["log"]
        else:
            edges = elements["edges"]
            pose = elements["pose"]
            img = elements["img"]
            focus = elements["focus"]
            fps = elements["fps"]
            results = elements["actions"]
            distance = elements["distance"]
            box = elements["box"]

            # POSE
            theta = 90
            R = np.matrix([[1, 0, 0],
                           [0, math.cos(theta), -math.sin(theta)],
                           [0, math.sin(theta), math.cos(theta)]])
            pose = pose @ R
            for i, edge in enumerate(edges):
                self.lines[i].set_data((pose[[edge[0], edge[1]]]))

            # IMAGE
            self.image.set_data(img)

            # INFO
            if focus:
                self.focus.text = "FOCUS"
                self.focus.color = "green"
            else:
                self.focus.text = "NOT FOCUS"
                self.focus.color = "red"
            self.fps.text = "FPS: {:.2f}".format(fps)
            self.distance.text = "DIST: {:.2f}m".format(distance)

            m = max([_[0] for _ in results.values()]) if len(results) > 0 else 0
            for i, r in enumerate(results.keys()):
                score, requires_focus, requires_box = results[r]
                # Check if conditions are satisfied
                if score == m:
                    c1 = True if not requires_focus else focus
                    c2 = True if (requires_box is None) else (box == requires_box)
                    if c1 and c2:
                        color = "green"
                    else:
                        color = "orange"
                else:
                    color = "red"
                if r in self.actions.keys():
                    text = "{}: {:.2f}".format(r, score)
                    if requires_focus:
                        text += ' (0_0)'
                    if requires_box:
                        text += ' [ ]'
                    if requires_box is not None and not requires_box:
                        text += ' [X]'
                    self.actions[r].text = text
                else:
                    self.actions[r] = Text('', rotation=0, anchor_x="center", anchor_y="bottom", font_size=12)
                    self.b2.add(self.actions[r])
                self.actions[r].pos = 0.5, 0.7 - (0.1 * i)
                self.actions[r].color = color

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
