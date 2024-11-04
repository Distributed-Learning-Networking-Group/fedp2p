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

master_port=29601
echo $master_port
kill -9 `lsof -t -i:$master_port`
# 检查端口是否放行
cd ~/code/communication
/bin/bash ./check_port.sh tcp ${master_port}

# 解除限速
sudo /bin/bash ~/code/communication/rate_limit.sh --del
echo "Unlimit the upload and download bandwidth."

cd ~/code/communication/bt_ps/
# /home/whr/.conda/envs/bt/bin/python 6.formal.py --rank=$RANK --world_size=4 --master_addr=192.168.124.102 --model resnet50 --dataset cifar10 --num_epochs 1 --aggregate_frequency 261 --log_level INFO --num_evaluate_threads 1 --batch_size 64
NCCL_DEBUG=INFO NCCL_SOCKET_IFNAME=enp5s0f1 /home/whr/.conda/envs/bt/bin/python 5.resnet_cifar_ddp.py --rank=$RANK --world_size=3 --master_addr=192.168.124.102 --master_port=$master_port --model=resnet101 --dataset=cifar10 --num_epochs=100 --log_level=INFO --batch_size=512 --lr=0.01
# /home/whr/.conda/envs/bt/bin/python 5.2.lenet_mnist_ddp.py --rank=$RANK --world_size=3 --master_addr=192.168.124.102 --model resnet34 --dataset cifar10 --num_epochs 100 --aggregate_frequency 261 --log_level INFO --num_evaluate_threads 1 --batch_size 128 --lr 0.1

# /home/whr/.conda/envs/bt/bin/python 7.linear_own_ddp.py --rank=$RANK --world_size=3 --master_addr=192.168.124.102 --model resnet50 --dataset cifar10 --num_epochs 200 --aggregate_frequency 261 --log_level INFO --num_evaluate_threads 1 --batch_size 64
