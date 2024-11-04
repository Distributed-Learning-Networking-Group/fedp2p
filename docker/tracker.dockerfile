FROM golang:1.19.1

WORKDIR /go/src/tracker
COPY check_port.sh /go/src/tracker
COPY start_tracker.sh  /go/src/tracker

# necessary system tools
RUN apt update && apt install -y sudo iptables

# clone source code from github.
# It's unstable for Chinese users to access github.
# As this source code will probably not be updated, I will just put the source code and its binary file inside this image and push this image to dockerhub.
# RUN git clone https://github.com/whr819987540/tracker.git && git clone https://github.com/chihaya/chihaya.git
COPY ./bt_ps/thirdparty/tracker /go/src/tracker/

RUN cd tracker && go env -w GOPROXY=https://proxy.golang.com.cn,direct && bash build.sh

CMD ["bash", "start_tracker.sh"]

# TAG="1.0"
# REGISTRY="192.168.124.102:5000"
# REGISTRY="172.28.7.173:5000"
# cd docker
# docker build -f ./tracker.dockerfile -t btps_tracker:$TAG ../ && docker tag btps_tracker:$TAG btps_tracker:latest 
# docker tag btps_tracker:latest 819987540whr/btps_tracker:latest && docker push 819987540whr/btps_tracker:latest

# docker run -it --rm --name btps_tracker btps_tracker:latest
# docker run -it --rm --name btps_tracker btps_tracker:latest cat tracker/config.yaml
