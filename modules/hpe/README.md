# Human Pose Estimator
This module contains a fast version of [Yolov4](https://github.com/Tianxiaomo/pytorch-YOLOv4) for Human Detection and a fast version of a truncation-resistant 3D absolute Human Pose Estimator [MetrABS](https://github.com/isarandi/metrabs).
All steps are accelerated with [TensorRT](https://developer.nvidia.com/tensorrt).
Note that this implementation is completely independent of PyTorch and TensorFlow since all the modules are transformed into ONNX and then TensorRT engines.

## Installation
Since this module is accelerated with TensorRT engines that works only on the right environment, one can choose between two alternatives:
- Create a Conda Environment and build the engines as detailed below
- Build the Docker image contained in the Dockerfile and use the provided engines

<span style="color:red">*NOTE*</span>: The Docker option is strongly recommended, since it doesn't require the creation of the engines (that takes over an hour), and performances and engines creation may be different in another system.
Anyway, if you really plan to use this module, recreate the engines is the best option.

### Run with Docker
Install Docker on your system.
Then, build the image launching the following command in a terminal/prompt inside the root directory of the project:

`docker build -t ecub .`

Download the engines from [here](https://drive.google.com/file/d/1iN3pL7WLgW-Gusc8Ou2NpUtYpTEFAhhM/view?usp=sharing) and place them inside [modules/hpe/weights/engines/docker](modules/hpe/weights/engines/docker).
After that, you can test the installation launching the following command in a terminal/prompt inside the root directory of the project (replace _PATH_ with _%cd%_ in Windows or _{$pwd}_ on Ubuntu):

`docker run -it --rm --gpus=all -v "PATH":/home/ecub ecub:latest python modules/hpe/hpe.py`


### Engines creation
In the [setup](setup) directory there are some scripts which guide you in the following steps:
1. Download and extract the ONNX of Yolov4
2. Download and extract the BackBone of MetrABS with full signature and extract the weight and bias of the Heads
3. Extract the ONNX of the BackBone and the Heads
4. Create the ONNX of the function _image_transformation_, that performs a homography between Yolo and MetrABS
5. Create the TensorRT engines of Yolo, the BackBone, the _image_transformation_ function and the Heads

After that, you can test the installation with `python modules/hpe/hpe.py`.
