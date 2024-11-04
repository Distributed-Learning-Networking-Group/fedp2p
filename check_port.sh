#!/bin/bash

# 检查端口是否放行
result=$(sudo iptables -nL|grep "${1}"|grep "${2}"|grep "ACCEPT")
echo "sudo iptables -nL|grep "${1}"|grep "${2}"|grep "ACCEPT""
if [ -z "${result}" ]
then
    echo "port ${2} is closed"
    sudo iptables -I INPUT -p ${1} --dport ${2} -j ACCEPT
    echo "port ${2} has been open"
else
    echo "port ${2} is open"
fi
