# Environment Information

- Ubuntu 20.04.6 LTS
- Docker 24.0.7
- Python 3.7.16
- Golang 1.19.1
- 7 physical machines, equipped with 128GB RAM and 4070 GPU and linked by a 10Gbps Ethernet  network.

# Functions of each file

- 运行脚本
  - shell scripts that set running environment and run python programs
  - `./*.sh`
- 定义节点的配置文件
  - configs that define registry, fluentd, workdir, node hostnames, executor info and so on
  - `./auto-generate.config.yml`
- 定义fluentd日志格式
  - define format of log message from fluentd container
  - `./fluentd.conf`
- dock compose生成脚本
  - python scripts that generate configs for docker compose
  - `./auto-generate_docker-compose.yaml.py`
- conda env yaml
  - `./bt.yml`
- install docker
  - `docker/intsall_docker.sh`
- install nvidia docker2
  - `docker/install_nvidia-docker2.sh`
- make gpu available in docker swarm
  - `docker/make_gpu_available_in_docker_swarm.sh`
- build image
  - `docker/tracker.dockerfile docker/system.dockerfile docker/client.dockerfile docker/server.dockerfile`
- grpc
  - `bt_ps/grpc_base.py bt_ps/grpc_client.py bt_ps/grpc_pb2_grpc.py bt_ps/grpc_pb2.py bt_ps/grpc.proto bt_ps/grpc_server.py`
- client selection
  - `bt_ps/client_manager.py`
- python programs for clients, server and the compute node
  - `bt_ps/ps_param_param_epoch_cs_computation.py bt_ps/ps_param_param_epoch_cs_grpc.py`
- environment variables
  - `common.env`

# Installation

It's highly recommended to use *docker swarm* for simulation, while directly running on physical machines is also available.

First, install docker and nvidia-docker2 with our scripts.

```bash
bash install_docker.sh
bash install_nvidia-docker2.sh
```

Then make GPU available in docker swarm.

```bash
bash make_gpu_available_in_docker_swarm.sh
```

Config the IP address of the registry container and restart docker.

```bash
bash config_docker_registry.sh 172.28.7.173:5000
sudo systemctl daemon-reload && sudo systemctl restart docker
```

Create and run the registry container.

```bash
sudo iptables -I INPUT -p tcp --dport 5000 -j ACCEPT
docker run -d -p 5000:5000 --name registry --restart always registry
```

Pull the tracker image from dockerhub and build the base system image, upon which we create the server and client image.

```bash
docker pull 819987540whr/btps_tracker:latest
# see more build instructions in the corresponding dockerfile
docker build -f ./system.dockerfile -t btps_system:$TAG --network host ../ && docker tag btps_system:$TAG btps_system:latest
docker build -f ./server.dockerfile -t btps_server:$TAG --network host ../ && docker tag btps_server:$TAG btps_server:latest
docker build -f ./client.dockerfile -t btps_client:$TAG --network host ../ && docker tag btps_client:$TAG btps_client:latest
```

Then you need to push these local images, so that other worker nodes can directly pull and use these images to avoid duplicate building. For example:

```bash
# on the machine that builds the image
docker tag btps_server:latest $REGISTRY/btps_server:latest && docker push $REGISTRY/btps_server:latest
# on other worker nodes
docker pull $REGISTRY/btps_server:latest
```

Create a swarm and make all machines join it.

```bash
docker swarm init
# get join token by `docker swarm join-token worker`
```

Create a docker overlay network.

```bash
docker network create --driver overlay --subnet=10.11.0.0/16 --gateway=10.11.0.2 test_overlay_network
```

# Simulation

First, set the configs in `auto-generate.config.yml` that define registry, fluentd, workdir, node hostnames, executor info and so on. And run `auto-generate_docker-compose.yaml.py` to generate the docker compose file `auto-generate.docker-compose.yml`

Set your timezone, name of docker stack, debug information and world size (the number of server and participants in each round) in `common.env`.

Launch the simulation by `docker compose -f fluentd.docker-compose.yml up -d && docker stack deploy -c auto-generate.docker-compose.yml $YourStackName`.

Logs, including log message and tensorboard data are availble in the `logs/$DateTime` directory.

Run `docker compose -f fluentd.docker-compose.yml down && docker stack rm $YourStackName` to stop the simulation.