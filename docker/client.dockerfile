# run client code based on bt_system:latest.
FROM btps_system:latest

LABEL maintainer="whr819987540"
LABEL version="2.4"
LABEL description="This is the image for client in BTPS system based on btps_system:latest."
LABEL messages="modify python modules"

WORKDIR /app

# load source code.
# for deployment, use COPY.
# COPY . /app/
# for development, use volumes in docker compose to save time.

CMD ["bash", "run_ps_with_rate_limited_in_docker.sh"]

# TAG="2.4"
# REGISTRY="192.168.124.102:5000"
# REGISTRY="172.28.7.173:5000"
# PROXY="http://127.0.0.1:7890"
# cd docker
# docker build -f ./client.dockerfile -t btps_client:$TAG --network host --build-arg HTTP_PROXY=$PROXY --build-arg HTTPS_PROXY=$PROXY ../ && docker tag btps_client:$TAG btps_client:latest
# docker tag btps_client:latest $REGISTRY/btps_client:latest && docker push $REGISTRY/btps_client:latest
# docker pull $REGISTRY/btps_client:latest

# docker run -itd --name btps_client btps_client:latest bash
# docker run -itd --rm -v /home/whr/code/communication:/app --name btps_client btps_client:latest bash
# docker run -itd --rm --cap-add=NET_ADMIN -v /home/whr/code/communication:/app --name btps_client btps_client:latest bash
# docker run -it --rm --name btps_client btps_client:latest ls /app
