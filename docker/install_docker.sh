#!/bin/bash

# install docker
sudo apt update
sudo apt install -y software-properties-common curl
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
sudo apt install -y docker-ce docker-ce-cli containerd.io
sudo systemctl enable docker
sudo systemctl restart docker
# 加入docker用户组
USER=$(whoami)
sudo usermod -aG docker $USER
