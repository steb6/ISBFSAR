# docker run -it --rm --gpus=all -v "C:/Users/sberti/PycharmProjects/ISBFSAR":/home/ecub ecub:latest /bin/bash
# Base human Pipeline Image
# It starts from a TensorRT container and install python, opencv, pytorch
FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04
ENV CONDA_VERSION "py38_4.11.0"
ENV CONDA_MD5 718259965f234088d785cad1fbd7de03

ENV PYTHONDONTWRITEBYTECODE=true

RUN apt-get update && apt-get install -y --no-install-recommends wget bzip2 \
    && addgroup ecub \
    && useradd -ms /bin/bash ecub -g ecub \
    && wget --quiet https://repo.continuum.io/miniconda/Miniconda3-$CONDA_VERSION-Linux-x86_64.sh \
    && mv Miniconda3-$CONDA_VERSION-Linux-x86_64.sh miniconda.sh \
    && sh ./miniconda.sh -b -p /opt/conda \
    && ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh \
    && echo ". /opt/conda/etc/profile.d/conda.sh" >> /home/ecub/.bashrc \
    && echo "conda activate base" >> /home/ecub/.bashrc \
    && find /opt/conda/ -follow -type f -name '*.a' -delete \
    && find /opt/conda/ -follow -type f -name '*.pyc' -delete \
    && /opt/conda/bin/conda clean -afy \
    && chown -R ecub:ecub /opt/conda
#    && apt-get --purge -y autoremove wget bzip2
RUN /opt/conda/bin/conda install --yes --freeze-installed -c rapidsai -c nvidia -c pytorch -c conda-forge\
    cuml=22.08 \
    pytorch \
    torchvision \
    cudatoolkit=11.3 \
    && /opt/conda/bin/conda clean -afy \
    && find /opt/conda/ -follow -type f -name '*.a' -delete ; 2>/dev/null\
    && find /opt/conda/ -follow -type f -name '*.pyc' -delete ; 2>/dev/null\
    && find /opt/conda/ -follow -type f -name '*.js.map' -delete -delete ; 2>/dev/null\
    && find /opt/conda/lib/python*/site-packages/bokeh/server/static -follow -type f -name '*.js' ! -name '*.min.js' -delete ; 2>/dev/null

RUN /opt/conda/bin/pip install --upgrade setuptools pip
RUN /opt/conda/bin/pip install --upgrade nvidia-pyindex nvidia-tensorrt pycuda
RUN /opt/conda/bin/pip install opencv-python einops tqdm playsound pyrealsense2 vispy omegaconf scipy mediapipe timm

RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get install ffmpeg libsm6 libxext6 -y
RUN /opt/conda/bin/pip install opencv-python

#ENV LD_LIBRARY_PATH "$LD_LIBRARY_PATH:/usr/local/cuda-11.3/lib:/usr/local/cuda-11.3/lib64:/usr/local/cuda-11.3/compat"
ENV PATH="/opt/conda/bin:/opt/conda/condabin:${PATH}"
USER ecub:ecub
WORKDIR /home/ecub

ENV PYTHONPATH="/home/ecub"
ENV AM_I_IN_A_DOCKER_CONTAINER Yes