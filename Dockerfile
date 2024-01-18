FROM ubuntu:22.04

# Install linux packages
RUN apt-get update && \
    apt-get install -y sudo make git wget python3-pip curl libxrender1 libsm6 libxext6 xauth mesa-utils nginx

# Install miniconda
RUN curl -LO https://repo.anaconda.com/miniconda/Miniconda3-py38_23.1.0-1-Linux-x86_64.sh && \
    bash Miniconda3-py38_23.1.0-1-Linux-x86_64.sh -b -p /miniconda && \
    rm Miniconda3-py38_23.1.0-1-Linux-x86_64.sh && \
    export PATH=/miniconda/bin:$PATH && \
    conda update -y conda

# Install environement
COPY ./env/chesslink.yml /env/chesslink.yml

# Configure python environment
RUN mkdir -p /workspace/.conda_cache
RUN /miniconda/bin/conda config --append pkgs_dirs /workspace/.conda_cache
RUN /miniconda/bin/conda env create -f /env/chesslink.yml --prefix /env/chesslink/
RUN /miniconda/bin/conda init
#RUN /miniconda/bin/source ~/.profile
#RUN /miniconda/bin/source activate /env/chesslink

# Copy NGINX conf
COPY ./src/web_app/server.conf /etc/nginx/sites-available/default

# Run the Nginx server
CMD /usr/sbin/nginx -g 'daemon off;';conda init;conda activate /env/chesslink
