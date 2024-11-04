import sys
sys.path.append("/app/bt_ps")
sys.path.append("/app/bt_ps/thirdparty/FedScale")
print(sys.path)
import json
import asyncio
import grpc
import random
import signal
import torch
import os
import logging
import pickle
import queue
import threading
import torch.nn as nn
import psutil

from datetime import datetime
from typing import Union, List, Set
from math import ceil
from queue import Empty
from d2l import torch as d2l
from copy import deepcopy
from torch.utils.tensorboard import SummaryWriter

import grpc_pb2
import grpc_pb2_grpc

from p2p_server.utils import utils
from grpc_base import GrpcBase, MessageType
from fedscale.utils import dataset as fedscale_dataset
from p2p_server.rpc.rpc_client import TorrentCommunicationPyTorch, TorrentCommunicationGRPC
from client_manager import ClientManager

# 配置参数
TRAINING_TIMEOUT = 300 # 每轮训练的超时时间（秒）

class TimeAnalysis:
    def __init__(self, summary_writer:SummaryWriter, logger:logging.Logger):
        self.summary_writer = summary_writer
        self.logger = logger
        
        self.data = {}
        self.items = []
        self.total_indeed_epoch_time = 0
        self.total_simulated_epoch_time = 0
        self.watch("distribution")
        self.watch("calculation")
        self.watch("aggregation")
        
    def watch(self, name):
        if name in self.items:
            raise ValueError
        self.items.append(name)
        self.data[name] = {
            f"{name}_time_list": [],
            f"averate_local_{name}_time_list": [],
            # tmp in a round
            f"{name}_start_time_list": [],
            f"{name}_stop_time_list": [],
            f"local_{name}_time_list": [],
        }
    
    @staticmethod
    def period_time(start_time_list:List, stop_time_list:List):
        return (max(stop_time_list) - min(start_time_list)).total_seconds()
    
    @staticmethod
    def average(time_list:List):
        return sum(time_list) / len(time_list)
    
    def analyse_selection_messages(self, selection_messages:List, round:int, round_start_time:datetime, round_end_time:datetime):
        self.logger.debug(f"selection_messages: {selection_messages}")
        for name in self.items:
            start_time_list = []
            stop_time_list = []
            local_time_list = []
            for selection_message in selection_messages:
                start_time_list.append(selection_message[f"{name}_start_time"])
                stop_time_list.append(selection_message[f"{name}_stop_time"])
                local_time_list.append(selection_message[f"{name}_time"])
            self.data[name][f"{name}_start_time_list"] = start_time_list
            self.data[name][f"{name}_stop_time_list"] = stop_time_list
            self.data[name][f"local_{name}_time_list"] = local_time_list
            self.analyse(start_time_list, stop_time_list, local_time_list, name, round)
            self.logger.info(f"{name} start: {min(start_time_list)}, stop: {max(stop_time_list)}")
        
        distribution_time_list = torch.tensor(self.data['distribution']['distribution_time_list'])
        calculation_time_list = torch.tensor(self.data['calculation']['calculation_time_list'])
        
        indeed_epoch_time = (round_end_time-round_start_time).total_seconds()
        self.total_indeed_epoch_time += indeed_epoch_time
        self.logger.info(f"[{round}] indeed time using: {indeed_epoch_time}")
        self.summary_writer.add_scalar("indeed_epoch_time", indeed_epoch_time, round)

        # simulated_epoch_time = (max(self.data['aggregation']['aggregation_stop_time_list']) - min(self.data['calculation']['calculation_start_time_list'])).total_seconds() + min(self.data['distribution']['local_distribution_time_list'])
        simulated_epoch_time = (max(self.data['aggregation']['aggregation_stop_time_list']) - min(self.data['aggregation']['aggregation_start_time_list'])).total_seconds() + (distribution_time_list+calculation_time_list).min().item()
        self.total_simulated_epoch_time += simulated_epoch_time
        self.logger.info(f"[{round}] simulated time using: {simulated_epoch_time}")
        self.summary_writer.add_scalar("simulated_epoch_time", simulated_epoch_time, round)
    
    def analyse(self, start_time_list:List, stop_time_list:List, local_time_list:List, name:str, round:int):
        time = TimeAnalysis.period_time(start_time_list, stop_time_list)
        average_local_time = TimeAnalysis.average(local_time_list)
        self.data[name][f"{name}_time_list"].append(time)
        self.data[name][f"averate_local_{name}_time_list"].append(average_local_time)
        self.summary_writer.add_scalar(f"{name}", time, round)
        self.summary_writer.add_scalar(f"local {name}", average_local_time, round)
        self.logger.info(f"{name} time: {time}, averate local {name} time: {average_local_time}")

class GrpcService(grpc_pb2_grpc.BroadcastServiceServicer, GrpcBase):
    def __init__(self, args, logger:logging.Logger):
        super().__init__()
        self.args = args
        self.logger = logger
        
        self.REQUIRED_UPDATES = self.args.selected_clients_number # 每轮需要的模型更新数量 
        self.TOTAL_SELECTED_CLIENTS = ceil(self.args.selected_clients_number*self.args.over_commitment) # 需要建立连接的客户端数量
        
        # self.model = torch.randn(10*10**6)
        self.model = None
        self.lock = asyncio.Lock()

        # 客户端管理
        self.connected_clients = {}  # {client_id: stream}
        self.clients_id = {} # {stream: client_id}
        self.client_id_counter = 0

        # 当前训练轮次信息
        self.current_training = {
            "clients_selected": set(),
            "updates_received": 0,
            "model_updates": [],
            "clients_updated": set(),
            "selection_received": set(),
            "selection_messages": [],
            "terminate_sent": False,
            "round": 1,
            "lock": asyncio.Lock(),
            "round_start_time": 0,
            "round_end_time": 0,
            "simulation_received": [],
            "simulation_messages": [],
        }

        # 任务管理
        self.current_wait_task = None  # 当前等待任务

        # 轮次控制
        self.training_in_progress = False

        # 优雅关闭
        self.shutdown_event = asyncio.Event()

        # 注册信号处理器
        self._register_shutdown_handlers()
        
        # 消息处理
        # self.message_queue = asyncio.Queue()
        # asyncio.create_task(self.process_messages())
        
        self.init_params(self.args)
        self.time_analysis = TimeAnalysis(self.summary_writer, self.logger)
        self.create_evaluete_thread()

    # 网络流量统计
    def log_network_bytes(self, interface_name="eth0"):
        net_io = psutil.net_io_counters(pernic=True)
        if interface_name in net_io:
            bytes_sent = net_io[interface_name].bytes_sent / 1073741824 # GB
            bytes_recv = net_io[interface_name].bytes_recv / 1073741824 # GB
            # 每轮开始前统计，因此将round-1
            self.logger.info(f"net_io: {interface_name}, bytes_sent: {bytes_sent}GB, bytes_recv: {bytes_recv}GB, round # {self.current_training['round']-1}")
            self.summary_writer.add_scalar("network_bytes_sent (GB)", bytes_sent, self.current_training["round"]-1)
            self.summary_writer.add_scalar("network_bytes_recv (GB)", bytes_recv, self.current_training["round"]-1)
        else:
            raise ValueError(f"Interface {interface_name} not found.")

    def init_params(self, args):
        super().init_params(args)
        self.summary_writer = SummaryWriter(
            os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                "logs",
                os.getenv("DateTime")
            )
        )

    async def prepare_for_training(self):
        while True:
            if len(self.connected_clients) < self.TOTAL_SELECTED_CLIENTS:
                self.logger.info(f"当前连接的客户端不足。已连接: {len(self.connected_clients)}, 需要: {self.TOTAL_SELECTED_CLIENTS}")
                await asyncio.sleep(5)
            else:
                break
        
        # config
        config_dict, json_config_path = utils.get_updated_config_file(self.args.master_addr, 
            self.args.master_port, self.args.model, self.args.dataset
        )
        with open(json_config_path, 'w') as f:
            f.write(json.dumps(config_dict))
        config_message = grpc_pb2.ServerMessage(
            config_distribute=grpc_pb2.ConfigDistribute(
                config=GrpcBase._serialize(config_dict),
                timestamp=self._get_current_timestamp(),
            )
        )
        self.logger.info(f"向客户端 {list(self.connected_clients.keys())} 发送 config")
        await self._broadcast_message(config_message, list(self.connected_clients.keys()), "config")
        self.config = utils.to_namespace(config_dict)
        self.model_root_dir = self.config.model.ModelPath
        os.makedirs(self.model_root_dir, exist_ok=True)
        self.logger.info(f"config: {self.config}")

        # client profiles
        # In simulation mode, server sends each client's computation and communication capacity.
        if self.args.running_mode == utils.SIMULATION_MODE:
            profiles = utils.load_client_profile()
            # upload rate should range from 10 to 50 mbps
            feasible_profiles = []
            for profile in profiles.values():
                upload_rate = profile['communication'] / 1024
                download_rate = upload_rate
                if download_rate >= 20:
                    feasible_profiles.append(profile)

            self.client_profiles = []
            for index in random.sample(range(1, len(feasible_profiles)), 1+self.participants_number):
                self.client_profiles.append(feasible_profiles[index])
            # # BUG:
            # self.client_profiles = self.client_profiles[:8]
            # self.client_profiles[1]['communication'] = 50*1024 # kbps
            # self.client_profiles[2]['communication'] = 100*1024 # kbps
            # self.client_profiles[3]['communication'] = 50*1024 # kbps
            # self.client_profiles[4]['communication'] = 100*1024 # kbps
            # self.client_profiles[5]['communication'] = 50*1024 # kbps
            # self.client_profiles[6]['communication'] = 100*1024 # kbps
            # self.client_profiles[7]['communication'] = 50*1024 # kbps
            
            self.logger.info(f"client_profiles {len(self.client_profiles)}")

            client_profile_message = grpc_pb2.ServerMessage(
                client_profile_distribute=grpc_pb2.ClientProfileDistribute(
                    client_profiles=GrpcBase._serialize(self.client_profiles),
                    timestamp=self._get_current_timestamp(),
                )
            )
            self.logger.info(f"向客户端 {list(self.connected_clients.keys())} 发送 client profiles")
            await self._broadcast_message(client_profile_message, list(self.connected_clients.keys()), "client profiles")
            self.logger.info(f"len(self.client_profiles): {len(self.client_profiles)}")

            # server profile
            # 1000*1024kbit/s
            self.server_profile = self.client_profiles[0]
            if self.args.model == "resnet18" and self.args.dataset == "femnist":
                self.server_profile['communication'] = 500*1024 # kbps
            elif self.args.model == "resnet34" and self.args.dataset == "google_speech_commands":
                self.server_profile['communication'] = 1000*1024 # kbps
            elif self.args.model == "shufflenet_v2_x2_0" and self.args.dataset == "openimage":
                self.server_profile['communication'] = 300*1024 # kbps
            else:
                self.server_profile['communication'] = 1000*1024 # kbps
            # BUG：
            self.server_profile['communication'] = 300*1024 # kbps
            self.server_profile['computation'] = 0
            self.logger.info(f"server_profile {self.server_profile}")

            if self.args.rate_limit:
                utils.rate_limit(self.args.rank, self.server_profile, self.logger)

        if self.args.transfer_mode == utils.BTPS_TRANSFER_MODE:
            self.torrent = None
            self.path = None
            self.tc = TorrentCommunicationGRPC(self.args.rank, self.logger)

        # model
        # model = models.get_model(args.model, args.dataset, args.num_classes)
        # 注意修改yml而非命令行参数
        self.model = fedscale_dataset.load_model()
        # log path
        self.log_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.log_path = os.path.join(self.log_path, "logs", os.getenv("DateTime"))
        # load from checkpoint
        self.checkpoint_model_name = f"{self.args.model}_{self.args.dataset}_backup.pth"

        if os.path.exists(self.checkpoint_model_name):
            self.logger.info(f"load model from checkpoint: {self.checkpoint_model_name}")
            self.model.load_state_dict(torch.load(self.checkpoint_model_name))
            model_size = os.path.getsize(self.checkpoint_model_name)
        else:
            torch.save(self.model.state_dict(), os.path.join(self.log_path, self.checkpoint_model_name))
            model_size = os.path.getsize(os.path.join(self.log_path, self.checkpoint_model_name))
        self.logger.info(f"model_size {model_size}")

        self.checkpoint_cm_name = f"{self.args.model}_{self.args.dataset}_cm.backup"
        self.client_manager = ClientManager(self.args, self.logger, self.participants_number, model_size)
        # 从bt_ps目录加载client_manager
        if os.path.exists(self.checkpoint_cm_name):
            with open(self.checkpoint_cm_name, "rb") as f:
                self.client_manager = pickle.load(f)
                self.logger.info(f"load client_manager from checkpoint: {self.checkpoint_cm_name}")
        
        self.test_dataloader = fedscale_dataset.get_test_dataloader(self.logger)
        self.loss_func = nn.CrossEntropyLoss()
        
        self.training_sets = fedscale_dataset.get_train_datasets(self.logger)
        self.client_manager.init_client_info(self.training_sets)

    async def _broadcast_message(self, payload:Union[object, List[object]], client_list:List[int], log_msg:str):
        """
        payload: If paylaod is a single object, send the same paylaod to each client.
            If it's a list, its length must equal to that of client_list.
            Besides, paylaod must be json serializable.
        client_list
        """
        tasks = []
        if isinstance(payload, list):
            if len(payload) != len(client_list):
                raise ValueError
            payload_list = payload
            for client_id, payload in zip(client_list, payload_list):
                stream = self.connected_clients.get(client_id)
                if stream:
                    try:
                        tasks.append(stream.write(payload))
                    except Exception as e:
                        self.logger.exception(f"向客户端 {client_id} 发送消息时出错: {e}")
        else:
            for client_id in client_list:
                stream = self.connected_clients.get(client_id)
                if stream:
                    try:
                        tasks.append(stream.write(payload))
                    except Exception as e:
                        self.logger.exception(f"向客户端 {client_id} 发送消息时出错: {e}")

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for idx, result in enumerate(results):
                if isinstance(result, Exception):
                    self.logger.exception(f"向客户端 {client_list[idx]} 发送 {log_msg}, 发生异常: {result}")
                    # 这里可以根据需要进行进一步的异常处理，如记录日志、清理资源等
                else:
                    self.logger.debug(f"向客户端 {client_list[idx]} 发送 {log_msg}，结果: {result}")
            return results

    async def Communicate(self, request_iterator, context):
        # 处理客户端的消息，client -> server
        try:
            async for client_message in request_iterator:
                await self.process_client_message(client_message, context)
        except asyncio.CancelledError as e:
            self.logger.exception(f"客户端 {self.clients_id[context]} 已断开连接: {e}")
        except grpc.aio.AioRpcError as e:
            async with self.lock:
                client_id = self.clients_id[context]
            self.logger.exception(f"与客户端 {self.clients_id[context]} 的通信发生错误: {e}")
        except Exception as e:
            async with self.lock:
                client_id = self.clients_id[context]
            self.logger.exception(f"处理客户端 {self.clients_id[context]} 的消息时出错: {e}")
        finally:
            # 移除客户端
            async with self.lock:
                if self.shutdown_event.is_set():
                    return
                client_id = self.clients_id[context]
                self.logger.info(self.connected_clients)
                self.logger.info(f"{dir(context)}")
                del self.connected_clients[client_id], self.clients_id[context]
                self.logger.info(f"客户端 {client_id} 已断开连接。")

    async def process_client_message(self, message, context):
        """根据消息类型处理来自客户端的消息"""
        # self.logger.info(message)
        if message.HasField(MessageType.ClientLogin.value):
            async with self.lock:
                client_id = message.client_login.rank
                self.connected_clients[client_id] = context
                self.clients_id[context] = client_id
                self.logger.info(f"客户端 {client_id} 已连接。")
        elif message.HasField(MessageType.ModelAggregate.value):
            message = message.model_aggregate
            await self.handle_model_update(message)
        elif message.HasField(MessageType.SelectionMessage.value):
            message = message.selection_message
            await self.handle_selection_message(message)
        elif message.HasField(MessageType.SimulationMessage.value):
            message = message.simulation_message
            await self.handle_simulation_message(message)
        else:
            self.logger.info(f"收到未知消息类型来自客户端 {client_id}。")

    async def handle_simulation_message(self, message):
        # 收到所有已连接客户端的模拟信息
        simulation_message = GrpcBase._deserialize(message.data)
        simulation_message['time'] = simulation_message['distribution_time'] + simulation_message['calculation_time']
        rank = message.rank
        client_id = message.client_id
        simulation_message['client_id'] = client_id
        async with self.current_training["lock"]:
            if rank not in self.current_training["clients_selected"]:
                self.logger.info(f"收到来自未选中客户端 {rank} 的模拟信息。")
                return
            elif rank in self.current_training["simulation_received"]:
                self.logger.info(f"收到来自客户端 {rank} 的重复模拟信息。")
                return
            self.current_training["simulation_received"].append(rank)
            self.current_training["simulation_messages"].append(simulation_message)
            self.logger.info(f"收到来自客户端 {rank} 的模拟信息 {len(self.current_training['simulation_received'])} / {self.TOTAL_SELECTED_CLIENTS}")
            
            if len(self.current_training["simulation_received"]) >= self.TOTAL_SELECTED_CLIENTS:
                self.logger.info(f"已收到所有客户端的模拟信息: {self.current_training['simulation_received']}, {self.current_training['simulation_messages']}")
                # 修正等待时间
                time_list = [simulation_message['time'] for simulation_message in self.current_training["simulation_messages"]]
                time_list = torch.tensor(time_list)
                self.logger.info(f"time_list: {time_list}")
                time_list = (time_list - time_list.min()).tolist()
                self.logger.info(f"time_list: {time_list}")
                clients_list = []
                message_list = []
                for idx, time in enumerate(time_list):
                    client_id = self.current_training["simulation_messages"][idx]['client_id']
                    clients_list.append(client_id)
                    message = grpc_pb2.ServerMessage(
                        simulation_message=grpc_pb2.SimulationMessage(
                            data=GrpcBase._serialize({"sleep_time": time}),
                            rank=self.current_training["simulation_received"][idx],
                            client_id=self.current_training["simulation_messages"][idx]['client_id'],
                        )
                    )
                    message_list.append(message)
                self.logger.info(f"向客户端 {clients_list} 发送 simulation sleep time")
                await self._broadcast_message(message_list, clients_list, "simulation sleep time")

    async def handle_selection_message(self, message):
        """处理来自客户端的选择消息"""
        async with self.current_training["lock"]:
            selection_message = GrpcBase._deserialize(message.data)
            client_rank = selection_message['rank']
            if client_rank not in self.current_training["clients_selected"]:
                self.logger.info(f"收到来自未选中客户端 {client_rank} 的统计消息。")
                return
            elif client_rank not in self.current_training["clients_updated"]:
                self.logger.info(f"收到未更新模型参数的客户端 {client_rank} 的统计消息。")
                return
            elif client_rank in self.current_training["selection_received"]:
                self.logger.info(f"收到来自客户端 {client_rank} 的重复统计消息。")
                return
            
            self.current_training["selection_received"].add(client_rank)
            self.current_training["selection_messages"].append(selection_message)
            
            self.logger.info(f"收到来自客户端 {client_rank} 的统计消息。总选择数: {len(self.current_training['clients_updated'])}/{self.REQUIRED_UPDATES}")

            # 检查是否已收到足够的统计信息
            if len(self.current_training["selection_messages"]) >= self.REQUIRED_UPDATES:
                self.logger.info("已收到足够的选择，开始新的训练轮次。")
                
                # 时间分析
                self.time_analysis.analyse_selection_messages(
                    self.current_training["selection_messages"], 
                    self.current_training['round'], 
                    self.current_training['round_start_time'],
                    self.current_training['round_end_time']
                )
                # evaluate check
                await self.evaluate()
                
                await self.reset_current_training()
                if self.current_training["round"] < self.args.num_epochs:
                    asyncio.create_task(self.start_new_training_round())
                else:
                    self.logger.info("达到最大训练轮数，关闭服务器。")
                    await self.shutdown_server()

    async def evaluate(self):
        # evaluate
        self.logger.info("evaluate")
        self.logger.info(f"len(evaluate_task_queue)={self.evaluate_task_queue.qsize()}")
        
        epoch = self.current_training["round"]
        model = deepcopy(self.model)
        
        if epoch % self.args.test_interval == 1 or self.args.test_interval == 1:
            self.evaluate_task_queue.put(
                (model, self.device, epoch, 0, self.test_dataloader, self.args, self.loss_func)
            )
            self.logger.info(f"len(evaluate_task_queue)={self.evaluate_task_queue.qsize()}")

            if epoch != 1:
                # result = evaluate(deepcopy(model), device, epoch, 0, test_dataloader, args, loss_func)
                result, eval_epoch = self.evaluate_result_queue.get()
                self.logger.info(f"evaluate result: {result}")
                self.logger.info(f"[epoch: {eval_epoch}] acc_1: {result[0]} acc_5: {result[1]} loss: {result[2]}")
                self.summary_writer.add_scalar("test_acc_1", result[0], eval_epoch)
                self.summary_writer.add_scalar("test_acc_5", result[1], eval_epoch)
                self.summary_writer.add_scalar("test_loss", result[2], eval_epoch)
                self.summary_writer.add_scalar("training_time_test_acc_1", result[0], int(self.time_analysis.total_simulated_epoch_time))
                self.summary_writer.add_scalar("training_time_test_acc_5", result[1], int(self.time_analysis.total_simulated_epoch_time))
                self.summary_writer.add_scalar("training_time_test_loss", result[2], int(self.time_analysis.total_simulated_epoch_time))

    async def handle_model_update(self, message):
        """处理来自客户端的模型更新"""
        async with self.current_training["lock"]:
            client_rank = message.rank
            if client_rank not in self.current_training["clients_selected"]:
                self.logger.info(f"收到来自未选中客户端 {client_rank} 的模型更新。")
                return
            elif client_rank in self.current_training["clients_updated"]:
                if message.round_number == self.current_training["round"]:
                    self.logger.info(f"收到来自客户端 {client_rank} 的重复模型更新。")
                else:
                    self.logger.info(f"收到来自客户端 {client_rank} 的过期模型更新。")
                return
            elif self.current_training["terminate_sent"]:
                self.logger.info(f"收到来自客户端 {client_rank} 的模型更新，但该轮聚合已结束。")
                return
            
            # 反序列化模型
            try:
                state_dict = GrpcBase._deserialize(message.state_dict)
                self.current_training["model_updates"].append({
                    "model_update": state_dict,
                    "dataset_length": message.dataset_length,
                })
                self.current_training["updates_received"] += 1
                self.current_training["clients_updated"].add(client_rank)
                self.logger.info(f"收到来自客户端 {client_rank} 的模型更新。总更新数: {self.current_training['updates_received']}/{self.REQUIRED_UPDATES}")
            except Exception as e:
                self.logger.exception(f"反序列化客户端 {client_rank} 的模型更新时出错: {e}")
                return

            # 检查是否已收到足够的更新
            if (self.current_training["updates_received"] >= self.REQUIRED_UPDATES and 
                not self.current_training.get("terminate_sent", False)):
                self.current_training['round_end_time'] = self._get_current_datetime()
                self.logger.info("已收到足够的模型更新，开始聚合并通知剩余客户端。")
                self.current_training["terminate_sent"] = True
                asyncio.create_task(self.terminate_and_aggregate())

    async def start_new_training_round(self):
        """开始新的训练轮次"""
        # server -> client
        async with self.current_training["lock"]:
            if self.current_wait_task and not self.current_wait_task.done():
                self.logger.info("正在取消之前的等待任务。")
                self.current_wait_task.cancel()
                try:
                    await self.current_wait_task
                except asyncio.CancelledError as e:
                    self.logger.info(f"之前的等待任务已取消: {e}")

            if len(self.connected_clients) < self.TOTAL_SELECTED_CLIENTS:
                self.logger.info(f"当前连接的客户端不足。已连接: {len(self.connected_clients)}, 需要: {self.TOTAL_SELECTED_CLIENTS}")
                await asyncio.sleep(10)  # 等待10秒后重试
                asyncio.create_task(self.start_new_training_round())
                return
            
            # 网络流量
            self.log_network_bytes()
            self.current_training['round_start_time'] = self._get_current_datetime()
            
            round = self.current_training["round"]
            # 选择客户端所要模拟的rank
            selected_clients = self.client_manager.select_clients(round, self.logger, self.current_training["selection_messages"])
            self.current_training["selection_messages"] = []
            selected_clients_rank = torch.nonzero(selected_clients).flatten()
            # # BUG
            # selected_clients_rank = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])
            self.logger.info(f"[{round}] selected_clients_rank: {selected_clients_rank}")
            
            self.current_training["clients_selected"] = selected_clients_rank

            # distribute simulated clients' rank
            client_rank_distribute = grpc_pb2.ServerMessage(
                client_rank_distribute=grpc_pb2.ClientRankDistribute(
                    simulated_ranks=selected_clients_rank.tolist(), # 将客户端-模拟rank映射发送给客户端
                    round_number=round,
                )
            )
            self.logger.debug(f"send time:{self._get_current_datetime()}")
            self.logger.info(f"向客户端 {list(self.connected_clients.keys())} 发送 round # {round}, client rank")
            await self._broadcast_message(client_rank_distribute, list(self.connected_clients.keys()), f"round # {round}, client rank")

            # distribute global model
            if self.args.transfer_mode == utils.BTPS_TRANSFER_MODE:
                if self.torrent is not None:
                    self.tc.stop_seeding(self.torrent)
                    os.remove(self.path)
                self.path = os.path.join(self.model_root_dir,  f"{round}.pth")
                torch.save(self.model.state_dict(), self.path)
                distribution_start_time = self._get_current_timestamp()
                self.torrent = await self.tc.bt_broadcast(self.path, distribution_start_time, round, self.connected_clients)
            elif self.args.transfer_mode == utils.PS_TRANSFER_MODE:
                serialized_state_dict = GrpcBase._serialize(self.model.state_dict())
                model_distribute = grpc_pb2.ServerMessage(
                    model_distribute=grpc_pb2.ModelDistribute(
                        state_dict=serialized_state_dict,
                        timestamp=self._get_current_timestamp(),
                        round_number=round,
                        simulated_ranks=selected_clients_rank.tolist(),
                    )
                )
                self.logger.debug(f"send time:{self._get_current_datetime()}")
                self.logger.info(f"向客户端 {list(self.connected_clients.keys())} 发送 round # {round}, model")
                await self._broadcast_message(model_distribute, list(self.connected_clients.keys()), f"round # {round}, model")

            # aggregate happens in coroutines

            # 启动等待训练轮次完成的任务
            self.current_wait_task = asyncio.create_task(self.wait_for_training_round())

    async def wait_for_training_round(self):
        """等待训练轮次完成或超时"""
        try:
            await asyncio.sleep(TRAINING_TIMEOUT)
            async with self.current_training["lock"]:
                if self.current_training["updates_received"] >= self.REQUIRED_UPDATES:
                    self.logger.info("训练轮次在超时前已完成。")
                    return
                self.logger.info(f"训练轮次超时。已收到 {self.current_training['updates_received']} 个更新。")
                if not self.current_training.get("terminate_sent", False):
                    self.current_training["terminate_sent"] = True
                    asyncio.create_task(self.terminate_and_aggregate(reason="Timeout"))
        except asyncio.CancelledError as e:
            self.logger.info(f"等待训练轮次的任务已取消: {e}")
            raise

    async def terminate_and_aggregate(self, reason:str=None):
        """终止剩余客户端并聚合模型"""
        async with self.current_training["lock"]:
            results = await asyncio.gather(
                self.terminate_remaining_clients(),
                self.aggregate_models(),
                return_exceptions=True
            )
            for idx, result in enumerate(results):
                if isinstance(result, Exception):
                    self.logger.exception(f"任务 {idx + 1} 发生异常: {result}")
                    # 这里可以根据需要进行进一步的异常处理，如记录日志、清理资源等
                else:
                    self.logger.info(f"任务 {idx + 1} 成功，结果: {result}")
            
            # 区分超时与正常结束
            # 超时时，需要先终止客户端，然后重启下一轮
            if reason is not None and reason == "Timeout":
                await self.reset_current_training()
                if self.current_training["round"] < self.args.num_epochs:
                    asyncio.create_task(self.start_new_training_round())
                else:
                    self.logger.info("达到最大训练轮数，关闭服务器。")
                    await self.shutdown_server()

    async def terminate_remaining_clients(self):
        """向尚未提交更新的客户端发送终止消息"""
        try:
            self.logger.info("terminate_remaining_clients")
            selected_clients = self.current_training["clients_selected"].tolist()
            del selected_clients[0] # server
            selected_clients = set(selected_clients)
            clients_updated = self.current_training["clients_updated"]
            clients_to_terminate = selected_clients - clients_updated
            
            selected_clients = self.current_training["clients_selected"].tolist()
            # 找到client_id，进而找到对应的stream
            client_id_to_terminate = [selected_clients.index(client_rank) for client_rank in clients_to_terminate]
            # selected_clients: [0, 3477, 5331, 6671, 7829, 7942, 8353, 9736], 
            # clients_updated: {7942, 6671, 5331, 3477, 7829}, 
            # clients_to_terminate: {9736, 8353}, 
            # client_id_to_terminate: [7, 6]
            self.logger.info(f"selected_clients: {selected_clients}, clients_updated: {clients_updated}, clients_to_terminate: {clients_to_terminate}, client_id_to_terminate: {client_id_to_terminate}" )
            
            round_terminate_message = grpc_pb2.ServerMessage(
                round_terminate=grpc_pb2.RoundTerminate(
                    round_number=self.current_training["round"],
                    timestamp=self._get_current_timestamp(),
                )
            )
            round = self.current_training["round"]
            
            self.logger.info(f"向客户端 {client_id_to_terminate} 发送 round # {round}, terminate")
            await self._broadcast_message(round_terminate_message, client_id_to_terminate, f"round # {round}, terminate")
        except Exception as e:
            self.logger.exception(f"终止剩余客户端时出错: {e}")
            
        # for client_id in clients_to_terminate:
        #     stream = self.connected_clients.get   (client_id)
        #     if stream:
        #         try:
        #             await stream.write(round_terminate_message)
        #             self.logger.info(f"已向客户端 {client_id} 发送终止消息。")
        #         except Exception as e:
        #             self.logger.exception(f"向客户端 {client_id} 发送终止消息时出错: {e}")

    async def aggregate_models(self):
        """聚合来自客户端的模型更新"""
        if not self.current_training["model_updates"]:
            self.logger.info("没有模型更新可聚合。")
            return

        num_models = len(self.current_training["model_updates"])
        self.logger.info(f"正在聚合 {num_models} 个模型更新。")

        global_state_dict = dict()
        if self.args.gradient_policy == utils.AggregateType.FEDAVG_STRATEGY.value:
            total_length = 0
            
            for data in self.current_training["model_updates"]:
                dataset_length = data["dataset_length"]
                total_length += dataset_length
                state_dict = data["model_update"]
                self.logger.info(f"dataset_length: {dataset_length}")
                
                for name, params in state_dict.items():
                    if name not in global_state_dict:
                        global_state_dict[name] = params * dataset_length
                    else:
                        global_state_dict[name] += (params * dataset_length)
            
            for name, params in global_state_dict.items():
                global_state_dict[name] = params / total_length

            self.model.load_state_dict(global_state_dict)
            self.logger.info(f"全局模型已更新。")
            self.backup_model_client_manager()
            self.logger.info(f"{os.path.join(self.log_path, self.checkpoint_model_name)}")
        elif self.args.gradient_policy == utils.AggregateType.FEDPROX_STRATEGY.value:
            pass

    def backup_model_client_manager(self):
        """备份模型和客户端管理器"""
        torch.save(self.model.state_dict(), os.path.join(self.log_path, self.checkpoint_model_name))
        with open(os.path.join(self.log_path, self.checkpoint_cm_name), "wb") as f:
            pickle.dump(self.client_manager, f)

    async def reset_current_training(self):
        """重置当前训练轮次的信息"""
        self.current_training["clients_selected"] = set()
        self.current_training["updates_received"] = 0
        self.current_training["model_updates"] = []
        self.current_training["clients_updated"] = set()
        self.current_training["selection_received"] = set()
        # self.current_training["selection_messages"] = [] # 完成客户端选择后手动重置
        self.current_training["terminate_sent"] = False
        self.current_training["round"] += 1
        # self.current_training["lock"] = asyncio.Lock() # 无需重置
        # self.current_training["round_start_time"] = 0 # 赋值时重置
        # self.current_training["round_end_time"] = 0 # 赋值时重置
        self.current_training["simulation_received"] = []
        self.current_training["simulation_messages"] = []
        
        self.logger.info(f"已重置训练轮次。当前轮次: {self.current_training['round']}")

    def _register_shutdown_handlers(self):
        """注册信号处理器，以便优雅关闭服务器"""
        for signame in {'SIGINT', 'SIGTERM'}:
            asyncio.get_event_loop().add_signal_handler(
                getattr(signal, signame),
                lambda: asyncio.create_task(self.shutdown_server())
            )

    async def shutdown_server(self):
        """优雅地关闭服务器，通知所有客户端并关闭连接"""
        self.logger.info("服务器正在关闭，向所有客户端发送终止消息。")
        training_terminate = grpc_pb2.ServerMessage(
            training_terminate=grpc_pb2.TrainingTerminate(
                timestamp=self._get_current_timestamp(),
            )
        )
        await asyncio.gather(
            *[self.send_termination(client_id, training_terminate) 
                for client_id in self.connected_clients.keys()],
            return_exceptions=True
        )

        # 取消等待任务
        if self.current_wait_task and not self.current_wait_task.done():
            self.logger.info("正在取消当前的等待任务。")
            self.current_wait_task.cancel()
            try:
                await self.current_wait_task
            except asyncio.CancelledError as e:
                self.logger.info(f"等待任务已取消: {e}")

        self.connected_clients.clear()
        self.logger.info("所有客户端已被通知并关闭连接。关闭服务器。")
        self.shutdown_event.set()

    async def send_termination(self, client_id, training_terminate):
        """发送终止消息并关闭客户端连接"""
        stream = self.connected_clients.get(client_id)
        if stream:
            try:
                await stream.write(training_terminate)
                self.logger.info(f"已向客户端 {client_id} 发送终止消息并关闭连接。")
            except Exception as e:
                self.logger.exception(f"向客户端 {client_id} 发送终止消息时出错: {e}")

    def create_evaluete_thread(self):
        self.evaluate_task_queue = queue.Queue(maxsize=1)
        self.evaluate_result_queue = queue.Queue(maxsize=1)
        self.evaluate_thread = threading.Thread(
            target=evaluate_handler, 
            args=(
                self.evaluate_task_queue, 
                self.evaluate_result_queue,
                self.logger
            )
        )
        self.evaluate_thread.start()
        self.logger.info("start evaluate thread")

def evaluate(model: torch.nn.Module, device, epoch, pwd_iter, data_loader, args, loss_func=None):
    # debug_param = next(model.parameters())
    # print("evaluate param:", debug_param[0][0])

    if isinstance(model, torch.nn.Module):
        model = model.to(device)
        model.eval()

    metrics = d2l.Accumulator(4)
    with torch.no_grad():
        for x, y in data_loader:
            if args.dataset == "google_speech_commands":
                x = torch.unsqueeze(x, 1)
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            metrics.add(
                # d2l.accuracy(y_hat, y), 
                utils.top_K_accuracy_pytorch(y_hat, y, 1),
                utils.top_K_accuracy_pytorch(y_hat, y, 5),
                loss_func(y_hat, y),
                d2l.size(y)
            )
    # print(f"[epoch: {epoch} iter: {pwd_iter}] acc: {metrics[0] / metrics[2]} loss: {metrics[1] / metrics[2]}")

    return metrics[0] / metrics[3], metrics[1] / metrics[3], metrics[2] / metrics[3]


def evaluate_handler(task_queue:queue.Queue, result_queue:queue.Queue, logger):
    while True:
        try:
            task = task_queue.get(timeout=30)
        except Empty as e:
            logger.debug(f"evaluate_handler {e}")
        else:
            # logger.info(f"evaluate params: {task}")
            result = evaluate(*task)
            result_queue.put((result, task[2]))

def create_server(args, logger):
    listen_addr = f"{args.master_addr}:{args.master_port}"
    server = grpc.aio.server(
        options=[
            ('grpc.max_send_message_length', 500 * 1024 * 1024),
            ('grpc.max_receive_message_length', 500 * 1024 * 1024),
        ]
    )
    broadcast_service = GrpcService(args, logger)
    grpc_pb2_grpc.add_BroadcastServiceServicer_to_server(broadcast_service, server)
    server.add_insecure_port(listen_addr)
    return server, broadcast_service

async def serve():
    listen_addr = '[::]:50051'
    grpc_server, grpc_service = create_server(listen_addr)
    await grpc_server.start()
    print(f"gRPC 服务器已启动，监听 {listen_addr}")
    # grpc_service

    # 等待服务器关闭事件
    await grpc_service.shutdown_event.wait()

    # 关闭服务器
    await grpc_server.stop(5)
    print("gRPC 服务器已关闭。")

if __name__ == '__main__':
    try:
        asyncio.run(serve())
    except KeyboardInterrupt:
        print("捕获到 KeyboardInterrupt，退出。")