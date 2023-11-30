#!/bin/bash
# After creating Docker container, this script is run from inside Docker container
# to install Conda environment and Pre-commit.
# It needs to access to volumes mounted in Docker container.

REPO_PATH="/workspace/ChessLink"
CONDA_ENV_NAME="jma_test"
CONDA_PKGS_CACHE="/workspace/.conda_cache"


# Setup Conda packages cache
if [ ! -z "$CONDA_PKGS_CACHE" ]
then
    echo ""
    echo "-- Using the folder $CONDA_PKGS_CACHE as cache for miniconda packages --"
    echo ""
    mkdir -p $CONDA_PKGS_CACHE
    conda config --append pkgs_dirs $CONDA_PKGS_CACHE
fi

# Install Conda environment
if [ ! -d $REPO_PATH/env/$CONDA_ENV_NAME ]; then
    echo ""
    echo "-- Installing Conda environment $CONDA_ENV_NAME --"
    echo ""
    conda env create -f $REPO_PATH/env/$CONDA_ENV_NAME.yml --prefix $REPO_PATH/env/$CONDA_ENV_NAME/
    conda init
    source ~/.profile
    source activate $REPO_PATH/env/$CONDA_ENV_NAME/

else
    echo ""
    echo "-- Conda env already exist, just initializing it --"
    echo ""
    conda init
    source ~/.profile
fi

# Fix stupid kornia bug (https://github.com/kornia/kornia/issues/2410)
sed -i '336s/.*/            if True:/' ./env/jma_test/lib/python3.10/site-packages/kornia/augmentation/container/video.py
# Also make sure noise is not constant over time
sed -i '191i \                            continue' ./env/jma_test/lib/python3.10/site-packages/kornia/augmentation/container/video.py
sed -i '191i \                        if k == "gaussian_noise" and (isinstance(module, (K.RandomGaussianNoise))):' ./env/jma_test/lib/python3.10/site-packages/kornia/augmentation/container/video.py


# Activate Conda environment automatically
echo "conda activate $REPO_PATH/env/$CONDA_ENV_NAME" >> ~/.bashrc
echo "cd $REPO_PATH" >> ~/.bashrc

# apt installs
sudo apt-get -y install screen
