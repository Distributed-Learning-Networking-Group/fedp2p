#!/bin/bash

if [ $# -eq 0 ]; then
    echo "No argument is provided."
    exit 1
fi

for arg in "$@"
do if [ "$arg" == "-h" ]; then
        echo "Config the addr of the local docker repository."
        echo "Usage: bash config_docker_registry.sh ADDR"
        echo "      -h print help message."
        exit 0
    fi
done

# all machines
sudo apt update && sudo apt-get update && sudo install -y jq
CONFIG=$(cat /etc/docker/daemon.json)
UPDATED_CONFIG=$(echo "$CONFIG" | jq --arg addr "$1" '.["insecure-registries"] = [$addr] ')

echo $UPDATED_CONFIG
echo $UPDATED_CONFIG > daemon.json.tmp
sudo cp daemon.json.tmp /etc/docker/daemon.json
rm daemon.json.tmp

sudo systemctl restart docker
