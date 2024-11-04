"""
    automatically generate docker-compose.yml to run in simulation mode
"""

import os
import yaml
from datetime import datetime
from pprint import pprint

def get_datetime_str():
    return datetime.strftime(datetime.now(), "%Y_%m_%d_%H_%M_%S")

def load_from_yml(config: str = "auto-generate.config.yml"):
    path = os.path.abspath(__file__)
    path = os.path.dirname(path)
    path = os.path.join(path, config)
    with open(path, "r") as f:
        res = yaml.load(f, Loader=yaml.FullLoader)

    pprint(res)
    return res


def convert_to_bytes(size_str):
    units = {"k": 1024, "m": 1024**2, "g": 1024**3, "t": 1024**4}
    size_str = size_str.lower()
    if size_str[-1].isdigit():  # 如果没有单位，默认为字节
        return int(size_str)
    else:
        return int(size_str[:-1]) * units[size_str[-1]]


if __name__ == "__main__":
    pwd = os.path.dirname(os.path.abspath(__file__))
    os.chdir(pwd)

    config = load_from_yml()
  
    # update datatime str (log dir)
    datetime_str = get_datetime_str()
    common_config = dict()
    with open("common.env","r") as f:
        for line in f.readlines():
            key, value = line.strip().split("=")
            common_config[key] = value
        # update
        common_config["DateTime"] = datetime_str
    with open("common.env","w") as f:
      for key, value in common_config.items():
          f.write(f"{key}={value}\n")
    # create log dir
    log_dir = f"logs/{datetime_str}"
    os.makedirs(log_dir, exist_ok=True)
    # make fluentd's user (systemd-network) can access and write the log dir
    os.chmod(log_dir, 0o777)

    # config fluentd
    fluentd = f"""version: '3.8'
services:
  fluentd:
    image: fluentd:latest
    container_name: fluentd
    volumes:
      - ./fluentd.conf:/fluentd/etc/fluentd.conf
      - ./logs/{datetime_str}:/fluentd/logs
    environment:
      - FLUENTD_CONF=fluentd.conf
    # As illustrated in https://stackoverflow.com/questions/45346005/cant-log-from-fluentd-logdriver-using-service-name-in-compose,
    # and https://github.com/moby/moby/issues/20370#issuecomment-185229591, 
    # the fluentd container should and must be accessed through the host network.
    network_mode: host
    ports:
      - 24224:24224/tcp"""
    with open("fluentd.docker-compose.yml", "w") as f:
      f.write(fluentd)

    
    # registry addr
    registry_addr = config['physical_machine']['registry']['addr']
    fluentd_addr = config['physical_machine']['fluentd']['addr']

    content = ""
    LF = "\n"

    # version
    content += "version: '3.8'\n"
    content += LF

    # comment on running commands
    content += "# python auto-generate_docker-compose.yaml.py\n"
    content += "# NTOE: before running, make sure that all previous containers are removed.\n"
    content += "# docker compose -f fluentd.docker-compose.yml up -d && docker stack deploy -c auto-generate.docker-compose.yml test\n"
    content += "# docker compose -f fluentd.docker-compose.yml down && docker stack rm test\n"
    content += "# docker service logs -f -t test_server\n"
    content += "# docker service logs -f -t test_client\n"
    content += "# docker service logs -f -t test_tracker\n"
    content += '# docker exec test_server.1.lduh19scc0wewvtesgkp1clu4 /opt/conda/envs/bt/bin/python -c "import torch;print(torch.cuda.is_available())"\n'
    content += LF

    # shared settings
    tmpfs_mount_point = config['virtual_machine']['tmpfs']['target']
    tmpfs_mount_size = convert_to_bytes(config['virtual_machine']['tmpfs']['size'])
    content += f"""x-shared-settings: &shared-settings
  networks:
    - test_overlay_network
  volumes:
    - {config['physical_machine']['workdir']}:{config['virtual_machine']['workdir']}
    - type: tmpfs
      target: {tmpfs_mount_point}
      tmpfs:
        size: {tmpfs_mount_size}
  env_file:
    - common.env"""
    content += LF
    content += LF

    # services
    services = []

    # server
    server = f"""services:
  server:
    image: {registry_addr}/btps_server:latest
    <<: *shared-settings
    cap_add:
      - NET_ADMIN
    depends_on:
      - tracker
      - tracker_redis
    logging:
      driver: fluentd
      options:
        fluentd-address: "{fluentd_addr}"
        fluentd-async-connect: "true"
        tag: "server"
    environment:
      - RANK=0
      - device={config['virtual_machine']['server']['device']}
      - NVIDIA_VISIBLE_DEVICES=all
    deploy:
      endpoint_mode: dnsrr
      restart_policy:
        condition: "none"
      placement:
        constraints: [node.hostname == {config['virtual_machine']['server']['hostname']}]"""
    services.append(server)
    
    # tracker
    tracker = f"""  tracker:
    image: whr8199/btps_tracker:latest
    <<: *shared-settings
    cap_add:
      - NET_ADMIN
    depends_on:
      - tracker_redis
    logging:
      driver: json-file
      options:
        max-size: "10m"
        max-file: "3"
    deploy:
      endpoint_mode: dnsrr
      placement:
        constraints: [node.hostname == {config['virtual_machine']['tracker']['hostname']}]"""
    services.append(tracker)

    # tracker_redis
    tracker_redis = f"""  tracker_redis:
    image: redis:alpine3.18
    <<: *shared-settings
    deploy:
      endpoint_mode: dnsrr
      placement:
        constraints: [node.hostname == {config['virtual_machine']['tracker_redis']['hostname']}]"""
    services.append(tracker_redis)

    # clients
    RANK = 0
    for index, client_config in enumerate(config['virtual_machine']['client']):
        print(client_config)
        hostname = client_config['hostname']
        device = client_config['device']
        for i in range(client_config['num']):
            RANK += 1
            container_name = f"client_{RANK}_{hostname}"
            client = f"""  {container_name}:
    image: {registry_addr}/btps_client:latest
    <<: *shared-settings
    cap_add:
      - NET_ADMIN
    depends_on:
      - server
      - tracker
      - tracker_redis
    logging:
      driver: fluentd
      options:
        fluentd-address: "{fluentd_addr}"
        fluentd-async-connect: "true"
        tag: "{container_name}"
    environment:
      - RANK={RANK}
      - device={device}
      - Hostname={hostname}
      - NVIDIA_VISIBLE_DEVICES=all
    deploy:
      endpoint_mode: dnsrr
      restart_policy:
        condition: "none"
      placement:
        constraints: [node.hostname == {hostname}]"""
            services.append(client)

    # for index, client_config in enumerate(config['virtual_machine']['client']):
        container_name = f"computation_node_{hostname}"
        computation_node = f"""  {container_name}:
    image: {registry_addr}/btps_client:latest
    <<: *shared-settings
    cap_add:
      - NET_ADMIN
    depends_on:
      - server
      - tracker
      - tracker_redis
    logging:
      driver: fluentd
      options:
        fluentd-address: "{fluentd_addr}"
        fluentd-async-connect: "true"
        tag: "{container_name}"
    command: ['bash', 'run_ps_computation_node_with_rate_limited_in_docker.sh']
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - device={device}
    deploy:
      endpoint_mode: dnsrr
      restart_policy:
        condition: "on-failure"
      placement:
        constraints: [node.hostname == {hostname}]"""
        services.append(computation_node)

    for service in services:
        content += service
        content += LF
        content += LF

    # network
    content += f"""networks:
  test_overlay_network:
    external: true
"""

    content += LF

    with open("auto-generate.docker-compose.yml", "w") as f:
        f.write(content)

    # check WORLD_SIZE set in the config file and common.env
    path = os.path.abspath(__file__)
    path = os.path.dirname(path)
    path = os.path.join(path, "common.env")
    with open(path,"r") as f:
        for line in f.readlines():
            if "WORLD_SIZE" in line:
                WORLD_SIZE = int(line.strip().split("=")[-1])
                print(f"WORLD_SIZE: {WORLD_SIZE}")
                break
            
    assert RANK + 1 == WORLD_SIZE, "*" * 10 + f" WORLD_SIZSE in common.env should be {RANK+1}! " + "*" * 10
    print("*" * 10 + f" SUCCESS! " + "*" * 10)
