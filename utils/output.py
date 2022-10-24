import cv2
from vispy import app, scene, visuals
from vispy.scene.visuals import Text, Image
import numpy as np
import math


def get_color(value):
    if 0 <= value < 0.33:
        return "red"
    if 0.33 < value < 0.66:
        return "orange"
    if 0.66 < value <= 1:
        return "green"
    raise Exception("Wrong argument")


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
        elif x.text == '`':
            self.os = not self.os
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

        self.os = True

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
        # Actions (LABEL OF INFO)
        self.fsscore = Text('fs score', color='white', rotation=0, anchor_x="center", anchor_y="bottom",
                            font_size=12, pos=(5/8, 0.75))
        self.b2.add(self.fsscore)
        self.osscore = Text('os score', color='white', rotation=0, anchor_x="center", anchor_y="bottom",
                            font_size=12, pos=(7/8, 0.75))
        self.b2.add(self.osscore)
        self.fsscore = Text('rf', color='white', rotation=0, anchor_x="center", anchor_y="bottom",
                            font_size=12, pos=(7/16, 0.75))
        self.b2.add(self.fsscore)
        self.os_score = scene.visuals.Rectangle(center=(2, 2), color="white", border_color="white", height=0.1)
        self.b2.add(self.os_score)
        # Actions
        self.focuses = {}
        self.actions = {}
        self.values = {}

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
        self.desc_add = Text('ADD ACTION: add action_name [-focus]', color='white', rotation=0,
                             anchor_x="left",
                             anchor_y="bottom",
                             font_size=10, pos=(0.1, 0.9))
        self.desc_save = Text('SAVE: save', color='white', rotation=0, anchor_x="left",
                              anchor_y="bottom",
                              font_size=10, pos=(0.1, 0.8))
        self.desc_load = Text('LOAD: load', color='white', rotation=0, anchor_x="left",
                              anchor_y="bottom",
                              font_size=10, pos=(0.1, 0.7))
        self.desc_debug = Text('DEBUG: debug', color='white', rotation=0, anchor_x="left",
                               anchor_y="bottom",
                               font_size=10, pos=(0.1, 0.6))
        self.desc_remove = Text('REMOVE ACTION: remove action_name', color='white', rotation=0, anchor_x="left",
                                anchor_y="bottom",
                                font_size=10, pos=(0.1, 0.5))
        self.input_string = Text(self.input_text, color='purple', rotation=0, anchor_x="left", anchor_y="bottom",
                                 font_size=12, pos=(0.1, 0.3))
        self.log = Text('', color='orange', rotation=0, anchor_x="left", anchor_y="bottom",
                        font_size=12, pos=(0.1, 0.2))
        b4.add(self.desc_add)
        b4.add(self.desc_save)
        b4.add(self.desc_load)
        b4.add(self.desc_debug)
        b4.add(self.desc_remove)
        b4.add(self.input_string)
        b4.add(self.log)

    def on_timer(self, _):
        if not self.show:
            return
        # Check if there is something to show
        elements = self.input_queue.get()
        if not elements:
            return
        # Parse elements
        elements = elements
        if "ACK" in elements.keys():  # Just an ack flag
            return
        if "log" in elements.keys():
            self.log.text = elements["log"]

        fps = elements["fps"]
        img = elements["img"]

        bbox = elements["bbox"] if "bbox" in elements.keys() else None
        edges = elements["edges"] if "edges" in elements.keys() else None
        pose = elements["pose"] if "pose" in elements.keys() else None
        distance = elements["distance"] if "distance" in elements.keys() else None

        focus = elements["focus"] if "focus" in elements.keys() else None
        face_bbox = elements["face_bbox"] if "face_bbox" in elements.keys() else None

        results = elements["actions"] if "actions" in elements.keys() else None
        is_true = elements["is_true"] if "is_true" in elements.keys() else None
        requires_focus = elements["requires_focus"] if "requires_focus" in elements.keys() else None

        # POSE
        if pose is not None:  # IF pose is not None, edges is not None
            theta = 90
            R = np.matrix([[1, 0, 0],
                           [0, math.cos(theta), -math.sin(theta)],
                           [0, math.sin(theta), math.cos(theta)]])
            pose = pose @ R
            for i, edge in enumerate(edges):
                self.lines[i].set_data((pose[[edge[0], edge[1]]]),
                                       color="purple",
                                       edge_color="white")
        else:
            for i in range(len(self.lines)):
                self.lines[i].set_data(color="grey",
                                       edge_color="white")

        # IMAGE
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if bbox is not None:
            x1, x2, y1, y2 = bbox
            img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
        if face_bbox is not None:
            x1, y1, x2, y2 = face_bbox
            color = (255, 0, 0) if not focus else (0, 255, 0)
            img = cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        self.image.set_data(cv2.flip(img, 0))

        # INFO
        if focus:
            self.focus.text = "FOCUS"
            self.focus.color = "green"
        else:
            self.focus.text = "NOT FOC."
            self.focus.color = "red"
        self.fps.text = "FPS: {:.2f}".format(fps)
        self.distance.text = "DIST: {:.2f}m".format(distance) if distance is not None else "DIST:"
        # Actions
        m = max(results.values()) if len(results) > 0 else 0  # Just max
        for i, action in enumerate(results.keys()):
            score = results[action]
            if action in self.actions.keys():  # Action was already in SS
                text = action
                self.actions[action].text = text
                self.values[action].width = score*0.25
                self.actions[action].pos = (3/16, 0.6 - (0.1 * i))
                self.values[action].center = (4/8 + ((score*0.25) / 2), 0.6 - (0.1 * i))
                self.values[action].color = get_color(score)
                self.values[action].border_color = get_color(score)
                if action in self.focuses.keys():
                    self.focuses[action].color = 'red' if not focus else 'green'
                    self.focuses[action].border_color = 'red' if not focus else 'green'
            else:  # Action must be added in SS
                # Action label
                self.actions[action] = Text('', rotation=0, anchor_x="center", anchor_y="center", font_size=12,
                                            pos=(3/16, 0.6 - (0.1 * i)), color="white")
                self.b2.add(self.actions[action])
                # Few Shot Label
                self.values[action] = scene.visuals.Rectangle(
                    center=(4/8 + ((score*0.25) / 2), 0.6 - (0.1 * i)),
                    color=get_color(score), border_color=get_color(score), height=0.1,
                    width=score*0.25)
                self.b2.add(self.values[action])
                # Eye for focus
                if requires_focus[action]:
                    self.focuses[action] = scene.visuals.Rectangle(center=(7/16, 0.6 - (0.1 * i)),
                                                                   color='red' if not focus else 'green',
                                                                   border_color='red' if not focus else 'green',
                                                                   height=0.1, width=0.05)
                    self.b2.add(self.focuses[action])
            # Os score
            self.actions[action].color = "white"
            if score == m:
                self.os_score.width = is_true*0.25
                self.os_score.center = [(6/8) + ((is_true*0.25) / 2), 0.6 - (0.1 * i)]
                self.os_score.color = get_color(is_true)
                self.os_score.border_color = get_color(is_true)
                if is_true > 0.66:
                    if requires_focus[action]:
                        self.actions[action].color = "green" if focus else "orange"
                    else:
                        self.actions[action].color = "green"
        # Remove erased action (if any)
        to_remove = []
        for key in self.actions.keys():
            if key not in results.keys():
                to_remove.append(key)
        for key in to_remove:
            self.actions[key].parent = None
            self.values[key].parent = None
            self.actions.pop(key)
            self.values.pop(key)
            if key in self.focuses.keys():
                self.focuses[key].parent = None
                self.focuses.pop(key)
        if len(self.actions) == 0:
            self.os_score.center = (2, 2)

    def on_draw(self, event):
        pass
