#! /bin/bash
if [ -z "$NET_IF" ]; then
    NET_IF="enp5s0f1"
fi
if [ -z "$UPLOAD_RATE" ]; then
    UPLOAD_RATE="1016kbit"
fi
if [ -z "$DOWNLOAD_RATE" ]; then
    DOWNLOAD_RATE="10160kbit"
fi

# ING_HANDLE=ffff
# BURST_VALUE_SEND=$((5 * 1024 * 1024))
BURST_VALUE_SEND=$((5 * 1024 * 1024))
# BURST_VALUE_RECV=$((8 * 1024 * 1024))

if [[ $# -ge 1 ]]; then
    # # del previous rule
    # sudo tc qdisc del dev $NET_IF root
    # sudo tc qdisc del dev $NET_IF ingress
    # sudo tc qdisc del dev $NET_IF parent $ING_HANDLE:
    # echo "del previous ruls and exit."

    # del previous rule
    sudo tc qdisc del dev $NET_IF root
    sudo tc qdisc del dev $NET_IF ingress
else
    # # upload rate
    # sudo tc qdisc add dev $NET_IF root tbf rate $UPLOAD_RATE latency 1000ms burst $BURST_VALUE_SEND
    # # download rate
    # sudo tc qdisc add dev $NET_IF handle $ING_HANDLE: ingress
    # sudo tc filter add dev $NET_IF parent $ING_HANDLE: protocol ip prio 0 u32 \
    #     match ip src all \
    #     action police rate $DOWNLOAD_RATE burst $BURST_VALUE_RECV drop flowid :1

    echo "UPLOAD_RATE: $UPLOAD_RATE, DOWNLOAD_RATE: $DOWNLOAD_RATE, BURST_VALUE_SEND: $BURST_VALUE_SEND"
    # upload rate
    sudo tc qdisc add dev $NET_IF root tbf rate $UPLOAD_RATE latency 1000ms burst $BURST_VALUE_SEND
    # download rate
    sudo tc qdisc add dev ${NET_IF} ingress
    sudo tc filter add dev ${NET_IF} protocol ip ingress prio 2 u32 match ip dst 0.0.0.0/0 action police rate ${DOWNLOAD_RATE} burst ${DOWNLOAD_RATE}
fi
