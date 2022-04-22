# Base human Pipeline Image
# It starts from a TensorRT container and install python, opencv, pytorch
FROM nvcr.io/nvidia/tensorrt:22.03-py3
# Necessary environment variables
#ENV CONDA_VERSION "py38_4.11.0"
#ENV CONDA_MD5 718259965f234088d785cad1fbd7de03
#ENV PYTHONDONTWRITEBYTECODE=true

## Install Miniconda3
#RUN apt-get update && apt-get install -y --no-install-recommends wget bzip2 \
#    && addgroup human \
#    && useradd -ms /bin/bash human -g human \
#    && wget --quiet https://repo.continuum.io/miniconda/Miniconda3-$CONDA_VERSION-Linux-x86_64.sh \
#    && mv Miniconda3-$CONDA_VERSION-Linux-x86_64.sh miniconda.sh \
#    && sh ./miniconda.sh -b -p /opt/conda \
#    && ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh \
#    && echo ". /opt/conda/etc/profile.d/conda.sh" >> /home/human/.bashrc \
#    && echo "conda activate base" >> /home/human/.bashrc \
#    && find /opt/conda/ -follow -type f -name '*.a' -delete \
#    && find /opt/conda/ -follow -type f -name '*.pyc' -delete \
#    && /opt/conda/bin/conda clean -afy \
#    && chown -R human:human /opt/conda \
#    && apt-get --purge -y autoremove bzip2 wget
#
## Install PyTorch
#RUN /opt/conda/bin/conda install --yes --freeze-installed -c conda-forge -c pytorch \
#        pytorch \
#        torchvision \
#        torchaudio \
#        cudatoolkit=11.3 \
#    && /opt/conda/bin/conda clean -afy \
#    && find /opt/conda/ -follow -type f -name '*.a' -delete ; 2>/dev/null \
#    && find /opt/conda/ -follow -type f -name '*.pyc' -delete ; 2>/dev/null \
#    && find /opt/conda/ -follow -type f -name '*.js.map' -delete -delete ; 2>/dev/null \
#    && find /opt/conda/lib/python*/site-packages/bokeh/server/static -follow -type f -name '*.js' ! -name '*.min.js' -delete ; 2>/dev/null

# Install pytorch
RUN pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113

# Install OpenCV
RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get install ffmpeg libsm6 libxext6 -y
RUN pip install opencv-python

# Install ptgaze
RUN apt-get install build-essential -y
RUN pip install ptgaze

# Install pycuda /usr/local/cuda-10.2/compat/lib.real/libcuda.so.1
#RUN /opt/conda/bin/conda install --yes pycuda -c conda-forge
#RUN /opt/conda/bin/pip install --upgrade pip
#RUN wget https://www.lfd.uci.edu/~gohlke/pythonlibs/#pycuda
#RUN pip install pycuda-2021.1+cuda102-cp38-cp38-win_amd64.whl
#RUN /opt/conda/bin/pip install pycuda

#ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/cuda-10.2/compat/lib.real/"
# ENV PATH="/opt/conda/bin:/opt/conda/condabin:${PATH}"
#USER human:human
#WORKDIR /home/human
COPY . .