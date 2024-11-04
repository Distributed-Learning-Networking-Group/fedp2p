#!/bin/bash

TC=$(which tc)
TC="sudo ${TC}"
# NET_IF="enp5s0f1"
# UPLOAD_RATE="1016kbit"
# DOWNLOAD_RATE="1016kbit"
# # bytes
# BURST_VALUE_SEND="13004"

if [[ $# -ge 1 ]]; then
    $TC qdisc del dev $NET_IF root
    $TC qdisc del dev $NET_IF ingress
    echo "Unlimit the upload and download bandwidth."
else
    $TC qdisc add dev $NET_IF root tbf rate $UPLOAD_RATE latency 1000ms burst $BURST_VALUE_SEND
    $TC qdisc add dev ${NET_IF} ingress
    $TC filter add dev ${NET_IF} protocol ip ingress prio 2 u32 match ip dst 0.0.0.0/0 action police rate ${DOWNLOAD_RATE} burst ${DOWNLOAD_RATE}
    echo "Limit the upload and download bandwidth."
fi
