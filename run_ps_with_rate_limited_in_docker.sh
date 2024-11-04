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
if [ -z "$RANK" ]; then
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

nvidia-smi

# fedscale install
cd $WORKDIR/bt_ps/thirdparty/FedScale
source install.sh
bash ~/.bashrc
# $PYTHON -m pip install -e .

# install p2p_server
# cd $WORKDIR/bt_ps
# $PYTHON -m pip install -e .

# 打印环境变量
env

# 设置时区
ln -fs /usr/share/zoneinfo/$TimeZone /etc/localtime && \
dpkg-reconfigure -f noninteractive tzdata
# # 时钟校准
timedatectl status
# timedatectl set-ntp true
# ntpdate ntp.tuna.tsinghua.edu.cn

echo $master_port
kill -9 `lsof -t -i:$master_port`
kill -9 `lsof -t -i:42069`
kill -9 `lsof -t -i:42070`
# # 检查端口是否放行
# export PASSWD=$PASSWD
cd $WORKDIR
/bin/bash ./check_port.sh tcp ${master_port}
/bin/bash ./check_port.sh tcp 42069
/bin/bash ./check_port.sh udp 42069
/bin/bash ./check_port.sh tcp 42070

# 解除限速
/bin/bash $WORKDIR/docker_rate_limit.sh --del

# 限速
# /bin/bash $WORKDIR/docker_rate_limit.sh

# /home/whr/.conda/envs/bt/bin/python public-asgd.py --rank=$RANK --world_size=4 --master_addr=192.168.124.102 --master_port $master_port --model resnet50 --dataset mnist --log_level DEBUG --batch_size 512 --num_epochs 100
cd $WORKDIR/bt_ps
# /home/whr/.conda/envs/bt/bin/python ps.py --rank=$RANK --world_size=4 --master_addr=192.168.124.102 --master_port=$master_port --model=resnet50 --dataset=cifar10 --log_level=DEBUG --batch_size=512 --num_epochs=100 --lr=0.01
# /home/whr/.conda/envs/bt/bin/python ps.py --rank=$RANK --world_size=4 --master_addr=192.168.124.102 --master_port=$master_port --model=resnet50 --dataset=cifar10 --log_level=INFO --batch_size=512 --num_epochs=100 --lr=0.01

# # google_speech_commands
# $PYTHON $WORKDIR/bt_ps/ps_param_param_epoch_cs.py \
# --rank=$RANK --world_size=$WORLD_SIZE --master_addr=$master_addr --master_port=$master_port \
# --log_level=INFO  --seed=$RANK \
# --model=resnet34 --dataset=google_speech_commands --num_classes=35 --num_epochs=50000 --lr=0.04 --test_interval=10 --local_epoch=1 \
# --running_mode=simulation --transfer_mode=PS --rate_limit=True \
# --use_gpu=True \
# --client_selection=True --selected_clients_number=100 --client_selection_strategy=fedavg --beta=2.0 --gradient_policy=fedavg

# openimage
# $PYTHON $WORKDIR/bt_ps/ps_param_param_epoch_cs.py \
# --rank=$RANK --world_size=$WORLD_SIZE --master_addr=$master_addr --master_port=$master_port \
# --log_level=INFO  --seed=$RANK \
# --model=shufflenet_v2_x2_0 --dataset=openimage --num_epochs=50000 --lr=0.04 --test_interval=10 --local_epoch=1 \
# --running_mode=simulation --transfer_mode=BTPS --rate_limit=True \
# --use_gpu=True \
# --client_selection=True --selected_clients_number=10 --client_selection_strategy=fedp2p --beta=1 --gradient_policy=fedavg

# $PYTHON -m pip install fastapi uvicorn websockets
# if [ "${RANK}" -eq 0 ]; then
#     # exec $PYTHON ws_server.py \
#     exec $PYTHON ps_param_param_epoch_cs_grpc.py \
#     --rank=$RANK --world_size=$WORLD_SIZE --master_addr=$master_addr --master_port=$master_port \
#     --log_level=INFO  --seed=$RANK \
#     --model=shufflenet_v2_x2_0 --dataset=openimage --num_epochs=200000 --lr=0.04 --test_interval=10 --local_epoch=1 \
#     --running_mode=simulation --transfer_mode=PS --rate_limit=True \
#     --use_gpu=True \
#     --client_selection=True --selected_clients_number=5 --over_commitment 1.3 --client_selection_strategy=oort --beta=1 --gradient_policy=fedavg
# else
#     # exec $PYTHON ws_client.py \
#     exec $PYTHON ps_param_param_epoch_cs_grpc.py \
#     --rank=$RANK --world_size=$WORLD_SIZE --master_addr=$master_addr --master_port=$master_port \
#     --log_level=INFO  --seed=$RANK \
#     --model=shufflenet_v2_x2_0 --dataset=openimage --num_epochs=200000 --lr=0.04 --test_interval=10 --local_epoch=1 \
#     --running_mode=simulation --transfer_mode=PS --rate_limit=True \
#     --use_gpu=True \
#     --client_selection=True --selected_clients_number=50 --over_commitment 1.3 --client_selection_strategy=random --beta=1 --gradient_policy=fedavg
# fi
# $PYTHON ps_param_param_epoch_cs_grpc.py \
# --rank=$RANK --world_size=$WORLD_SIZE --master_addr=$master_addr --master_port=$master_port \
# --log_level=INFO --seed=$RANK \
# --model=resnet34 --dataset=google_speech_commands --num_epochs=200000 --lr=0.04 --test_interval=10 --local_epoch=1 \
# --running_mode=simulation --transfer_mode=PS --rate_limit=True \
# --use_gpu=True \
# --client_selection=True --selected_clients_number=50 --over_commitment 1.0 --client_selection_strategy=random --beta=1 --gradient_policy=fedavg

$PYTHON ps_param_param_epoch_cs_grpc_com.py \
--rank=$RANK --world_size=$WORLD_SIZE --master_addr=$master_addr --master_port=$master_port \
--log_level=INFO --seed=$RANK \
--model=resnet34 --dataset=google_speech_commands --num_epochs=200000 --lr=0.04 --test_interval=10 --local_epoch=1 \
--running_mode=simulation --transfer_mode=BTPS --rate_limit=True --quick_simulate=False \
--use_gpu=True \
--client_selection=True --selected_clients_number=50 --over_commitment=1.3 --client_selection_strategy=oort --beta=1 --gradient_policy=fedavg

# # femnist
# $PYTHON $WORKDIR/bt_ps/ps_param_param_epoch_cs.py \
# --rank=$RANK --world_size=$WORLD_SIZE --master_addr=$master_addr --master_port=$master_port \
# --log_level=INFO  --seed=$RANK \
# --model=resnet18 --dataset=femnist --num_epochs=50000 --lr=0.04 --test_interval=10 --local_epoch=1 \
# --running_mode=simulation --transfer_mode=PS --rate_limit=True \
# --use_gpu=True \
# --client_selection=True --selected_clients_number=100 --client_selection_strategy=fedavg --beta=1.0 --gradient_policy=fedavg


# $PYTHON $WORKDIR/bt_ps/ps_param_param_epoch_cs.py \
# --rank=$RANK --world_size=$WORLD_SIZE --master_addr=$master_addr --master_port=$master_port \
# --log_level=INFO  --seed=$RANK \
# --model=shufflenet_v2_x2_0 --dataset=openimage --num_epochs=50000 --lr=0.04 --test_interval=10 --local_epoch=10 \
# --running_mode=simulation --transfer_mode=BTPS --rate_limit=True \
# --use_gpu=True \
# --client_selection=True --selected_clients_number=50 --client_selection_strategy=oort --gradient_policy=fedavg



# --client_selection=True --selected_clients_number=50 --client_selection_strategy=fedavg --gradient_policy=fedprox

# $PYTHON $WORKDIR/bt_ps/ps_param_param_epoch_cs.py \
# --rank=$RANK --world_size=$WORLD_SIZE --master_addr=$master_addr --master_port=$master_port \
# --log_level=INFO  --seed=$RANK \
# --model=resnet18 --dataset=femnist --num_epochs=1000 --lr=0.04 --test_interval=5 \
# --running_mode=simulation --transfer_mode=PS --rate_limit=True \
# --use_gpu=True \
# --client_selection=True --selected_clients_number=25 --client_selection_strategy=oort

# --model=resnet34 --dataset=google_speech_commands --num_epochs=1000 --lr=0.04 --test_interval=1 \
# --model=resnet18 --dataset=femnist --num_epochs=1000 --lr=0.04 --test_interval=1 \
# --model=resnet34 --dataset=openimg --num_epochs=10000 --lr=0.04 --test_interval=20 \


# --client_selection False
# --client_selection=True --selected_clients_proportion=0.1 --client_selection_strategy=fedavg
# --client_selection=True --selected_clients_proportion=0.1 --client_selection_strategy=oort

# --client_selection False
# --model=cnn --dataset=mnist --batch_size=512 --num_epochs=20 --lr=0.0001 \
# --model=resnet18 --dataset=cifar10 --batch_size=512 --num_epochs=100 --lr=0.01 \
# --model=resnet34 --dataset=mnist --batch_size=50 --num_epochs=1000 --lr=0.01 --shard_size=150 \
# --client_selection=True --selected_clients_proportion=0.5
# --client_selection=True --selected_clients_proportion=0.1 --client_selection_strategy=fedavg
# --client_selection=True --selected_clients_proportion=0.1 --client_selection_strategy=oort

# 解除限速
/bin/bash $WORKDIR/docker_rate_limit.sh --del
