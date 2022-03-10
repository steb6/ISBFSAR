# MetrABS 16 fps
This repository contains all the steps necessary to run [MetrABS](https://github.com/isarandi/metrabs) at 16 fps with a GPU on Windows.

## Description
This is obtained by using a third party Human Detector [Yolov4](https://github.com/Tianxiaomo/pytorch-YOLOv4) and by accelerating with [TensorRT](https://developer.nvidia.com/tensorrt) the inference of both the detector and the BackBone.
The weight and bias of the final convolution are extracted and applied on the CPU.
Note that this implementation is completely independent of PyTorch and TensorFlow since all the modules are transformed into ONNX and then TensorRT engines.

## Preparation
In order to run the `metrabs_trt.py` script, some steps are needed
In the [setup](modules/hpe/setup) directory you can find five scripts which guide you in the following steps:
1. Download and extract the ONNX of Yolov4
2. Download and extract the BackBone of MetrABS with full signature and extract the weight and bias of the final convolution
3. Extract the ONNX of the BackBone
4. Create the ONNX of the function `image_transformation` (note: this function was part of the `tensorflow_addons` package, but since we want to avoid Tensorflow in deployment and since this function performs a lot of computation, I reimplemented it in PyTorch)
5. Create the TensorRT engines of Yolo, the BackBone and the `image_transformation` function

## Run
In order to make inference, just start the [hpe.py](modules/hpe/hpe.py) script. You can change the parameters in the [configuration file](utils/params.py).