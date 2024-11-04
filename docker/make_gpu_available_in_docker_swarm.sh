#!/bin/bash

# 修改 /etc/nvidia-container-runtime/config.toml 文件
# 取消注释 swarm-resource = "DOCKER_RESOURCE_GPU"
sudo sed -i '/#swarm-resource = "DOCKER_RESOURCE_GPU"/s/^#//g' /etc/nvidia-container-runtime/config.toml

# 提取 GPU UUID
# GPU_UUIDS=$(nvidia-smi -a | grep "GPU UUID" | awk '{print $4}' | awk -F '-' '{print $1"-"$2}')
GPU_UUIDS=$(nvidia-smi -a | grep "GPU UUID" | awk '{print $4}')

# 将GPU UUID都写入 /etc/docker/daemon.json
# 安装json文件写入工具
sudo apt update && sudo apt-get update && sudo install -y jq
# 准备 JSON 格式的 GPU 配置
GPU_RESOURCES=$(echo "$GPU_UUIDS" | awk '{print "NVIDIA-GPU="$1""}' | paste -sd, -)
# 读取现有的 /etc/docker/daemon.json 文件内容
CONFIG=$(cat /etc/docker/daemon.json)
# 使用 jq 工具添加 GPU UUID 到配置中(需预先安装 jq)
UPDATED_CONFIG=$(echo "$CONFIG" | jq '.["default-runtime"] = "nvidia"')
UPDATED_CONFIG=$(echo "$UPDATED_CONFIG" | jq --arg gpu_resources "$GPU_RESOURCES" '.["node-generic-resources"] += [$gpu_resources]')

echo $UPDATED_CONFIG
echo $UPDATED_CONFIG > daemon.json.tmp
sudo cp daemon.json.tmp /etc/docker/daemon.json
rm daemon.json.tmp

# 重启 Docker 服务
sudo systemctl restart docker
