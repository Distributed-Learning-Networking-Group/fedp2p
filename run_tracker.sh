#!/bin/bash
function check_port
{
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
}

check_port tcp 6880
check_port tcp 6969
check_port udp 6969

# run
if [ $# -eq 0 ]
then
# quiet
nohup ./tracker --config ./config.yaml >> tracker.log 2>&1 &
else
# debug
nohup ./tracker --config ./config.yaml >> tracker.log --debug 2>&1 &
fi