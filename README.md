# Interactive Skeleton Based Few Shot Action Recognition
This repository contains a fast implementation of a 3d truncation robust [human pose estimator module](https://github.com/isarandi/metrabs), a fast [few shot action recognition module](https://github.com/tobyperrett/trx) and a CLI to add, remove or modify new action.
If you are interested in the human pose estimator module take a look at [hpe](modules/hpe), if you are interested in the action recognition module take a look at [trx](modules/ar/trx).

## Create environment
Create a Conda environment and install the following packages:
1) `conda install -c conda-forge opencv`
2) `conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch`
3) `conda install -c conda-forge matplotlib`
4) `pip install tensorflow`
5) `conda install -c conda-forge scipy`

## Run
Simply run `python main.py`
