#!/bin/bash
# 移除所有状态为 Down 的节点

docker node ls --format "{{.ID}} {{.Status}}" | awk '$2=="Down"{print $1}' | xargs -I {} docker node demode {}
docker node ls --format "{{.ID}} {{.Status}}" | awk '$2=="Down"{print $1}' | xargs -I {} docker node rm {}
