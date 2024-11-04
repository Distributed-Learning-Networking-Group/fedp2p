#! /bin/bash

# 这个配置下, 发送: 速率是100mbps, 时延是100ms; 接收: 速率是100mbps, 时延是100ms
NET_IF=enp5s0f1
BURST_VALUE_RECV=$((512 * 1024)) # should be tuned per rate change.
RATE="100mbit"
DELAY=50ms
JITTER=0ms
LOSS_RATE=0 # percent, random drop
REORDER=0 #percent, out of order sent

ING_HANDLE=ffff

sudo tc qdisc del dev $NET_IF root 2>/dev/null
sudo tc qdisc del dev $NET_IF parent $ING_HANDLE: 2>/dev/null

if [[ $# -ge 1 ]]; then
   echo "del previous ruls and exit."
else
sudo tc qdisc add dev $NET_IF root netem \
   rate $RATE \
   loss random $LOSS_RATE \
   delay $DELAY

sudo tc qdisc add dev $NET_IF handle $ING_HANDLE: ingress

sudo tc filter add dev $NET_IF parent $ING_HANDLE: protocol ip prio 0 u32 \
   match ip src all\
   action police rate $RATE burst $BURST_VALUE_RECV drop flowid :1
fi

# #!/bin/bash

# # 设置网络接口名称，根据需要替换为你的网络接口
# NET_IF="enp5s0f1"

# # 网速限制为100mbit
# RATE="1000mbit"

# # 时延限制为100ms
# DELAY="100ms"


# if [[ $# -ge 1 ]]; then
#    sudo tc qdisc del dev $NET_IF root 2>/dev/null
#    echo "del previous ruls and exit."
# else
#    # 添加新的流量控制规则
#    sudo tc qdisc add dev $NET_IF root handle 1: netem rate $RATE delay $DELAY
#    echo "Network speed limited to $RATE with a delay of $DELAY on $NET_IF"
# fi
