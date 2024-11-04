#!/bin/bash
WORKDIR=/go/src/tracker
if [ -z "$TimeZone" ]; then
    TimeZone="Asia/Shanghai"
fi
if [ -z "$StackName" ]; then
    exit 1
else
    redis_addr=${StackName}_tracker_redis
    echo $redis_addr
    sed -i 's/redis:\/\/127.0.0.1/redis:\/\/'"$redis_addr"'/' $WORKDIR/tracker/config.yaml
fi

# 设置时区
ln -fs /usr/share/zoneinfo/$TimeZone /etc/localtime && \
dpkg-reconfigure -f noninteractive tzdata

# # 检查端口是否放行
# export PASSWD=$PASSWD
# /bin/bash ./check_port.sh tcp ${master_port}
# /bin/bash ./check_port.sh tcp 6880
# /bin/bash ./check_port.sh tcp 6969
# /bin/bash ./check_port.sh udp 6969

if [ -n "$DEBUG" ] && [[ "$DEBUG" == true ]] ; then
    echo "run in debug"
    cd $WORKDIR/tracker && ./tracker --config ./config.yaml --debug
else
    echo "not run in debug"
    cd $WORKDIR/tracker && ./tracker --config ./config.yaml
fi
