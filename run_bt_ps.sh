#!/bin/bash

# rank
IP=`ifconfig enp5s0f1|awk '/inet / {print $2}'`
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
    cd ~/code/communication/bt_ps/p2p_server/rpc/rpc_server
    source build.sh
    if [ $? -eq 0 ]
    then
        echo "rebuild"
    else
        echo "build go code error"
        exit
    fi
fi

# 清理进程
master_port=29600
echo $master_port
kill -9 `lsof -t -i:$master_port`
kill -9 `lsof -t -i:42069`
kill -9 `lsof -t -i:42070`

cd ~/code/communication/bt_ps/
# /home/whr/.conda/envs/bt/bin/python distributed.py --rank=$RANK --world_size=4 --master_addr=192.168.124.102 --model lenet --dataset mnist --num_epochs 100 --aggregate_frequency 261 --log_level INFO --num_evaluate_threads 1 --batch_size 128 --lr 0.1
/home/whr/.conda/envs/bt/bin/python distributed.py --rank=$RANK --world_size=4 --master_addr=192.168.124.102 --master_port $master_port --model resnet50 --dataset cifar10 --num_epochs=100 --aggregate_frequency 261 --log_level INFO --num_evaluate_threads 1 --batch_size 64 --lr 0.01

# /home/whr/.conda/envs/bt/bin/python 8.linear_own_distributed.py --rank=$RANK --world_size=4 --master_addr=192.168.124.102 --model resnet18 --dataset cifar10 --num_epochs 2 --aggregate_frequency 261 --log_level DEBUG --num_evaluate_threads 200 --batch_size 512
