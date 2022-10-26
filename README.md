# Interactive Open-Set Skeleton-Based One-Shot Action-Recognition

[![MIT License](https://img.shields.io/badge/license-MIT-green)](https://opensource.org/licenses/MIT)
[![GitHub stars](https://img.shields.io/github/stars/stefanoberti/ISBFSAR.svg?style=flat-square&logo=github&label=Stars&logoColor=white)](https://github.com/hysts/pytorch_mpiigaze)

The aim of this project is to provide an efficient pipeline for Action Recognition in Human Robot Interaction.

The whole 3D human pose is estimated and used to understand which action inside the support set the human is performing.
Action can be easily added or removed from the support set in any moment.
The Open-Set score confirms or rejects the Few-Shot prediction to avoid false positives.
The Mutual Gaze Constraint can be added to an action as additional filter.
![Our visualizer](assets/demo.gif)
## Modules
This repository contains different modules:
- [hpe](modules/hpe): a Human Pose Estimation module, that is an accelerated version of [MetrABS](https://github.com/isarandi/metrabs) with the usage of Nvidia TensorRT
- [ar](modules/ar): an Action Recognition module, that is the evolution of the implementation of our paper [One-Shot Open-Set Skeleton-Based Action Recognition](https://arxiv.org/abs/2209.04288) _by [Stefano Berti](https://github.com/stefanoberti), [Andrea Rosasco](https://github.com/andrearosasco), [Michele Colledanchise](https://github.com/miccol), [Lorenzo Natale](https://github.com/lornat75) with [Istituto Italiano di Tecnologia](https://iit.it)_
- [focus](modules/focus): a Focus Detection Module, that uses [MPIIGaze](https://github.com/hysts/pytorch_mpiigaze) to do Gaze Estimation and checks for an intersection with the camera

## Installation

The program is divided into two parts:
- [source.py](source.py) runs on the host machine, it connects to the RealSense (or webcam), it provides frames to [main.py](main.py), it visualizes the results with the [VISPYVisualizer](utils/output.py)
- [main.py](main.py) runs either in a Conda environment or in a Docker, it is responsible for all the computation part.

![](assets/Immagine1.png)

Since the hpe modules is accelerated with TensorRT engines that requires to be built on the target machine, we provide the engines build over the Dockerfile, that allows for a fast installation.
Check [here](modules/hpe/README.md) the instruction to install the Human Pose Estimation module.


### Run with Docker
Follow the instruction inside the README.md of every module: [hpe](modules/hpe/README.md), [ar](modules/ar/README.md), and [focus](modules/focus/README.md).
Install [Vispy](https://github.com/vispy/vispy) and [pyrealsense2](https://pypi.org/project/pyrealsense2/) and build the Docker image with:

`docker build -t ecub .`

To run, start two separate processes:

`python manager.py`
`python source.py`

Launch the main script with the following command (replace _PATH_ with _%cd%_ in Windows or _{$pwd}_ on Ubuntu):

`docker run -it --rm --gpus=all -v "PATH":/home/ecub ecub:latest python main.py`
