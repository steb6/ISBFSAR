import copy
import multiprocessing
from multiprocessing import Queue, Process
from multiprocessing.managers import BaseManager
from typing import Dict, Union
from utils.input import RealSense
from utils.output import VISPYVisualizer
from utils.params import MainConfig


"""
This module manages the input and the output of the whole program.
It also manages the communication with the docker container.
Input: frames from RealSense (camera.read()) or commands fom VISPY (input_queue.get())
Output: frames from RealSense or commands fom VISPY (processes['src_to_sink'].put)
Input: elements for VISPY to visualize (processes['sink_to_src'].get())
Output: elements for VISPY to visualize (output_queue.send())
"""

if __name__ == '__main__':
    multiprocessing.current_process().name = 'Source'

    processes: Dict[str, Union[Queue, None]] = {'src_to_sink': None, 'sink_to_src': None}

    BaseManager.register('get_queue')
    manager = BaseManager(address=('localhost', 50000), authkey=b'abracadabra')
    manager.connect()

    for proc in processes:
        processes[proc] = manager.get_queue(proc)

    # Create input (camera)
    camera = RealSense(width=MainConfig().cam_width, height=MainConfig().cam_height, fps=60)

    # Create output (vispy)
    input_queue = Queue(1)
    output_queue = Queue(1)
    output_proc = Process(target=VISPYVisualizer.create_visualizer,
                          args=(output_queue, input_queue))
    output_proc.start()

    while True:
        _, rgb = camera.read()

        if not input_queue.empty():  # Vispy sent a command
            msg = input_queue.get()
            processes['src_to_sink'].put({'msg': copy.deepcopy(msg)})
        else:
            processes['src_to_sink'].put({'rgb': copy.deepcopy(rgb)})  # TODO MAYBE NOT INDENT
        ret = processes['sink_to_src'].get()
        if "log" in ret.keys():

        output_queue.put((ret,))
