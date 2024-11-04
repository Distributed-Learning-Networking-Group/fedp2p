#!/bin/bash
WORKDIR=~/code/communication
PASSWD="whr123456"
if [ -z "$PYTHON" ]; then
    PYTHON=/home/whr/.conda/envs/bt/bin/python
fi
NET_IF="enp5s0f1"
master_port=29601

IP=`ifconfig $NET_IF|awk '/inet / {print $2}'`
echo ${IP##*.}
declare -i LAST=${IP##*.}
echo $LAST
if [ $LAST -eq 102 ]
then
    RANK=0
else
    RANK=$(expr $LAST - 102 - 1)
fi
echo $RANK

if [ $RANK -eq 0 ]
then
    cd $WORKDIR/bt_ps/p2p_server/rpc/rpc_server
    /bin/bash build.sh
    if [ $? -eq 0 ]
    then
        echo "rebuild"
    else
        echo "build go code error"
        exit
    fi
fi

echo $master_port
kill -9 `lsof -t -i:$master_port`
kill -9 `lsof -t -i:42069`
kill -9 `lsof -t -i:42070`
# 检查端口是否放行
cd $WORKDIR
export PASSWD=$PASSWD
/bin/bash ./check_port.sh tcp ${master_port}
/bin/bash ./check_port.sh tcp 42069
/bin/bash ./check_port.sh udp 42069
/bin/bash ./check_port.sh tcp 42070

# 解除限速
/bin/bash $WORKDIR/rate_limit.sh --del
echo "Unlimit the upload and download bandwidth."

# # 限速
# /bin/bash $WORKDIR/rate_limit.sh
# echo "Limit the upload and download bandwidth."

# /home/whr/.conda/envs/bt/bin/python public-asgd.py --rank=$RANK --world_size=4 --master_addr=192.168.124.102 --master_port $master_port --model resnet50 --dataset mnist --log_level DEBUG --batch_size 512 --num_epochs 100
cd $WORKDIR/bt_ps
# /home/whr/.conda/envs/bt/bin/python ps.py --rank=$RANK --world_size=4 --master_addr=192.168.124.102 --master_port=$master_port --model=resnet50 --dataset=cifar10 --log_level=DEBUG --batch_size=512 --num_epochs=100 --lr=0.01
# /home/whr/.conda/envs/bt/bin/python ps.py --rank=$RANK --world_size=4 --master_addr=192.168.124.102 --master_port=$master_port --model=resnet50 --dataset=cifar10 --log_level=INFO --batch_size=512 --num_epochs=100 --lr=0.01
$PYTHON $WORKDIR/bt_ps/ps_param_param_epoch_cs.py \
--rank=$RANK --world_size=4 --master_addr=192.168.124.102 --master_port=$master_port \
--log_level=INFO --seed=$RANK \
--model=cnn --dataset=mnist --batch_size=512 --num_epochs=20 --lr=0.0001 \
--running_mode=standalone --transfer_mode=PS \
--use_gpu=True \
--client_selection False


# --client_selection False
# --model=cnn --dataset=mnist --batch_size=512 --num_epochs=20 --lr=0.0001 \
# --model=resnet18 --dataset=cifar10 --batch_size=512 --num_epochs=50 --lr=0.01 \
# --client_selection True --selected_clients_proportion 0.5

# 解除限速
/bin/bash $WORKDIR/rate_limit.sh --del
echo "Unlimit the upload and download bandwidth."
