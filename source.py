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

    processes: Dict[str, Union[Queue, None]] = {'source_human': None, 'human_sink': None}

    BaseManager.register('get_queue')
    manager = BaseManager(address=('localhost', 50000), authkey=b'abracadabra')
    manager.connect()

    for proc in processes:
        processes[proc] = manager.get_queue(proc)

    # Create input (camera)
    camera = RealSense(width=MainConfig().cam_width, height=MainConfig().cam_height, fps=60)

    # Create output (vispy)
    vispy_in_q = Queue(1)
    vispy_out_q = Queue(1)
    output_proc = Process(target=VISPYVisualizer.create_visualizer,
                          args=(vispy_out_q, vispy_in_q))
    output_proc.start()

    elems = {}
    while True:
        _, rgb = camera.read()

        # Prepare inference with rgb and optional command got from Vispy
        elems['rgb'] = copy.deepcopy(rgb)
        elems['msg'] = '' if vispy_in_q.empty() else copy.deepcopy(vispy_in_q.get())

        # Send to main
        processes['source_human'].put(elems)

        # Send results to visualizer
        vispy_out_q.put(processes['human_sink'].get())
