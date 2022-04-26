# Base human Pipeline Image
# It starts from a TensorRT container and install python, opencv, pytorch
FROM nvcr.io/nvidia/tensorrt:22.03-py3

# Install pytorch
RUN pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113

# Install OpenCV
RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get install ffmpeg libsm6 libxext6 -y
RUN pip install opencv-python

# Install ptgaze
#RUN apt-get install build-essential -y
#RUN pip install ptgaze


#ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/cuda-11.6/compat/lib.real/"
#USER human:human
#WORKDIR /home/human
COPY . .