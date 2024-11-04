# build the system with no source code involved.
FROM continuumio/anaconda3:2023.03-1

LABEL maintainer="whr819987540"
LABEL version="2.4"
LABEL description="This is the base image for the BTPS system."
LABEL messages="modify python modules"

WORKDIR /app

# necessary system tools
# wget is used to download go.
# tc in iproute2 is used to limit network bandwidth.
RUN apt update && apt install -y sudo wget iproute2 lsof iptables net-tools build-essential libsndfile1
# env variables
ENV PYTHON=/opt/conda/envs/bt/bin/python
ENV GO=/usr/local/go/bin/go

# create python virtual environment
RUN conda create -y --name bt python=3.7

# install python packages
# fedscale
COPY ./bt_ps/thirdparty/FedScale/environment.yml /root
RUN conda env update --name bt --file /root/environment.yml
RUN $PYTHON -m pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117 && $PYTHON -m pip install matplotlib d2l tensorboard flask && $PYTHON -m pip install fastapi gunicorn

# install go
# Chinese users may fail here and you should try more times.
RUN cd ~ && wget https://go.dev/dl/go1.19.1.linux-amd64.tar.gz && rm -rf /usr/local/go && tar -C /usr/local -xzf go1.19.1.linux-amd64.tar.gz

# CMD
CMD $PYTHON -c "import sys;print(sys.path);" && $GO version


# TAG="2.4"
# REGISTRY="192.168.124.102:5000"
# REGISTRY="172.28.7.173:5000"
# PROXY="http://127.0.0.1:7890"
# cd docker
# docker build -f ./system.dockerfile -t btps_system:$TAG --network host --build-arg HTTP_PROXY=$PROXY --build-arg HTTPS_PROXY=$PROXY ../ && docker tag btps_system:$TAG btps_system:latest 
# docker tag btps_system:latest $REGISTRY/btps_system:latest && docker push $REGISTRY/btps_system:latest

# docker run -it --rm --name btps_system btps_system:latest
# docker run -it --rm --name btps_system btps_system:latest ls /app
# docker run -itd --name btps_system btps_system:latest bash
# GPU + volume
# docker run -itd --name btps_system -v /home/whr/code/communication:/app --gpus all btps_system:1.0 bash 

# # install python packages
# $PYTHON -m pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
# $PYTHON -m pip install matplotlib d2l

# # install go
# wget https://go.dev/dl/go1.19.1.linux-amd64.tar.gz
# rm -rf /usr/local/go
# tar -C /usr/local -xzf go1.19.1.linux-amd64.tar.gz

# docker pull ufoym/deepo:latest
# docker run -itd --gpus "device=0" ufoym/deepo bash
# docker run -itd --gpus "device=0" --name test btps_system:latest bash
# docker run -it --gpus "device=0" --rm btps_system:latest nvidia-smi
# docker run -itd --gpus "device=0" --name test continuumio/anaconda3:2023.03-1 bash
# docker run -it --gpus "device=0" --rm continuumio/anaconda3:2023.03-1 bash
# docker run -it --gpus "device=0" --rm btps_system:latest /opt/conda/envs/bt/bin/python -c "import torch;print(torch.cuda.is_available())"

# $PYTHON -c "import sys;print(sys.path);" && $GO version