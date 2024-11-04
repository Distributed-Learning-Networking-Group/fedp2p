# run server code based on bt_system:latest.
FROM btps_system:latest

LABEL maintainer="whr819987540"
LABEL version="2.4"
LABEL description="This is the image for server in BTPS system based on btps_system:latest."
LABEL messages="modify python modules"

WORKDIR /app

# load source code.
# for deployment, use COPY.
# COPY . /app/
# for development, use volumes in docker compose to save time.

# server uses this port to communicate control messages by torch DDP.
EXPOSE 29601/tcp


CMD ["bash", "run_ps_with_rate_limited_in_docker.sh"]

# TAG="2.4"
# REGISTRY="192.168.124.102:5000"
# REGISTRY="172.28.7.173:5000"
# PROXY="http://127.0.0.1:7890"
# cd docker
# docker build -f ./server.dockerfile -t btps_server:$TAG --network host --build-arg HTTP_PROXY=$PROXY --build-arg HTTPS_PROXY=$PROXY ../ && docker tag btps_server:$TAG btps_server:latest
# docker tag btps_server:latest $REGISTRY/btps_server:latest && docker push $REGISTRY/btps_server:latest

# docker run -itd --name btps_server btps_server:latest bash
# docker run -it --rm --name btps_server btps_server:latest ls /app
# docker run -itd --rm --cap-add=NET_ADMIN -v /home/whr/code/communication:/app --shm-size=3G --name btps_server btps_server:latest bash

# docker exec test_server.1 ps -aux

# export PYTHON=/opt/conda/envs/bt/bin/python WORKDIR=/app NET_IF=eth0 master_addr=pc2 master_port=29601 RANK=0
