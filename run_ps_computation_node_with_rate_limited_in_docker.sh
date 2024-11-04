#!/bin/bash
if [ -z "$WORKDIR" ]; then
    WORKDIR=~/code/communication
    export WORKDIR="$WORKDIR"
fi
if [ -z "$PYTHON" ]; then
    PYTHON=/home/whr/.conda/envs/bt/bin/python
    export PYTHON="$PYTHON"
fi
if [ -z "$GO" ]; then
    exit 1
fi
if [ -z "$StackName" ]; then
    exit 1
else
    master_addr=${StackName}_server
    echo $master_addr
fi
if [ -z "$NET_IF" ]; then
    NET_IF="eth0"
    export NET_IF="$NET_IF"
fi
if [ -z "$master_port" ]; then
    master_port=29601
    export master_port="$master_port"
fi
# if [ -z "$UPLOAD_RATE" ]; then
#     UPLOAD_RATE="300mbit"
#     export UPLOAD_RATE="$UPLOAD_RATE"
# fi
# if [ -z "$DOWNLOAD_RATE" ]; then
#     DOWNLOAD_RATE="300mbit"
#     export DOWNLOAD_RATE="$DOWNLOAD_RATE"
# fi
if [ -z "$TimeZone" ]; then
    TimeZone="Asia/Shanghai"
fi

# fedscale install
cd $WORKDIR/bt_ps/thirdparty/FedScale
source install.sh
bash ~/.bashrc

nvidia-smi
env

# 设置时区
ln -fs /usr/share/zoneinfo/$TimeZone /etc/localtime && \
dpkg-reconfigure -f noninteractive tzdata

cd $WORKDIR/bt_ps
$PYTHON -m pip install gunicorn
$PYTHON ps_param_param_epoch_cs_computation.py \
--model=resnet34 --dataset=google_speech_commands
# /opt/conda/envs/bt/bin/gunicorn ps_param_param_epoch_cs_computation:app --bind 0.0.0.0:27500 --workers 4 --threads 2
# --model=resnet34 --dataset=google_speech_commands
# --model=resnet18 --dataset=femnist
# --model=shufflenet_v2_x2_0 --dataset=openimage
