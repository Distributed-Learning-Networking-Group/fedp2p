#!/bin/bash
cd ~/code/communication/ps/
NET_IF="enp5s0f1"
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

master_port=29601
echo $master_port
kill -9 `lsof -t -i:$master_port`
kill -9 `lsof -t -i:42069`
kill -9 `lsof -t -i:42070`
# 检查端口是否放行
cd ~/code/communication
/bin/bash ./check_port.sh tcp ${master_port}
/bin/bash ./check_port.sh tcp 42069
/bin/bash ./check_port.sh udp 42069
/bin/bash ./check_port.sh tcp 42070

# 解除限速
sudo /bin/bash ~/code/communication/latency_limit.sh --del
echo "Unlimit the upload and download bandwidth."

# 限速
sudo /bin/bash ~/code/communication/latency_limit.sh
echo "Limit the upload and download bandwidth."

# /home/whr/.conda/envs/bt/bin/python public-asgd.py --rank=$RANK --world_size=4 --master_addr=192.168.124.102 --master_port $master_port --model resnet50 --dataset mnist --log_level DEBUG --batch_size 512 --num_epochs 100
cd ~/code/communication/bt_ps
# /home/whr/.conda/envs/bt/bin/python ps.py --rank=$RANK --world_size=4 --master_addr=192.168.124.102 --master_port=$master_port --model=resnet50 --dataset=cifar10 --log_level=DEBUG --batch_size=512 --num_epochs=100 --lr=0.01
# /home/whr/.conda/envs/bt/bin/python ps.py --rank=$RANK --world_size=4 --master_addr=192.168.124.102 --master_port=$master_port --model=resnet50 --dataset=cifar10 --log_level=INFO --batch_size=512 --num_epochs=100 --lr=0.01
/home/whr/.conda/envs/bt/bin/python ~/code/communication/bt_ps/ps.py --rank=$RANK --world_size=4 --master_addr=192.168.124.102 --master_port=$master_port --model=resnet101 --dataset=cifar10 --log_level=INFO --batch_size=512 --num_epochs=10 --lr=0.01

# 解除限速
sudo /bin/bash ~/code/communication/latency_limit.sh --del
echo "Unlimit the upload and download bandwidth."
