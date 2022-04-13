from queue import Empty
from vispy import app, scene, visuals
from vispy.scene.visuals import Text, Image


class VISPYVisualizer:

    def printer(self, x):
        if x.text == '\b':
            self.input_text = self.input_text[:-1]
            self.log.text = ''
        elif x.text == '\r':
            self.output_queue.put(self.input_text)
            self.input_text = ''
            self.log.text = ''
        else:
            self.input_text += x.text
        self.input_string.text = self.input_text

    def __init__(self, input_queue, output_queue):

        self.input_queue = input_queue
        self.output_queue = output_queue

        self._timer = app.Timer('auto', connect=self.on_timer, start=True)
        self.input_text = ''

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
        b1.camera = scene.TurntableCamera(45, elevation=-90, azimuth=0, distance=2)
        self.lines = []
        Plot3D = scene.visuals.create_visual_node(visuals.LinePlotVisual)
        for _ in range(30):
            self.lines.append(Plot3D(
                [],
                width=2.0,
                color="green",
                edge_color="w",
                symbol="o",
                face_color=(0.2, 0.2, 1, 0.8),
                marker_size=1,
            ))
            b1.add(self.lines[_])

        # Info
        self.b2 = grid.add_view(row=0, col=1)
        self.b2.camera = scene.PanZoomCamera(rect=(0, 0, 1, 1))
        self.b2.border_color = (0.5, 0.5, 0.5, 1)
        self.fps = Text('FPS:', color='white', rotation=0, anchor_x="left", anchor_y="bottom",
                        font_size=18, pos=(0.5, 1))
        self.focus = Text('Non focus', color='green', rotation=0, anchor_x="left", anchor_y="bottom",
                          font_size=18, pos=(0, 1))
        self.b2.add(self.fps)
        self.b2.add(self.focus)
        self.actions = {}

        # Image
        b3 = grid.add_view(row=1, col=0)
        b3.camera = scene.PanZoomCamera(rect=(0, 0, 640, 480))
        b3.border_color = (0.5, 0.5, 0.5, 1)
        self.image = Image()
        # self.focus.pos = 20, 20
        b3.add(self.image)

        # Commands
        b4 = grid.add_view(row=1, col=1)
        b4.camera = scene.PanZoomCamera(rect=(0, 0, 1, 1))
        b4.border_color = (0.5, 0.5, 0.5, 1)
        self.desc = Text('Insert action', color='white', rotation=0, anchor_x="left", anchor_y="bottom",
                         font_size=18, pos=(0, 1))
        self.input_string = Text('', color='blue', rotation=0, anchor_x="left", anchor_y="bottom",
                                 font_size=18, pos=(0, 0.8))
        self.log = Text('', color='white', rotation=0, anchor_x="left", anchor_y="bottom",
                        font_size=18, pos=(0, 0.2))
        b4.add(self.desc)
        b4.add(self.input_string)
        b4.add(self.log)

    def on_timer(self, _):
        try:
            elements = self.input_queue.get_nowait()
        except Empty:
            return
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
            if results is not None:
                m = max([_[0] for _ in results.values()])
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
                        self.actions[r] = Text('', rotation=0, anchor_x="center", anchor_y="bottom", font_size=18)
                        self.b2.add(self.actions[r])
                    self.actions[r].pos = 0.5, 0.8 - (0.2 * i)
                    self.actions[r].color = color
                    i += 1

    def on_draw(self, event):
        pass
