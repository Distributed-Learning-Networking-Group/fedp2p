import sys
sys.path.append("/app/bt_ps")
sys.path.append("/app/bt_ps/thirdparty/FedScale")
print(sys.path)
import grpc
import random
import asyncio
import torch
import time
import logging
import os
import requests
import base64
import pytz
import torch.nn as nn

from datetime import datetime
from requests.exceptions import ConnectionError, ReadTimeout
from urllib3.exceptions import ProtocolError
from json.decoder import JSONDecodeError
from http.client import RemoteDisconnected
from grpc.aio import AioRpcError
from threading import Event
from fedscale.utils import dataset as fedscale_dataset
from fedscale.cloud.fllibs import *
from copy import deepcopy

import grpc_pb2
import grpc_pb2_grpc

from p2p_server.utils import utils
from grpc_base import GrpcBase, MessageType
from p2p_server.rpc.rpc_client import TorrentCommunicationGRPC

beijing_tz = pytz.timezone('Asia/Shanghai')
def timestamp_to_datetime(t:float):
    return datetime.fromtimestamp(t, beijing_tz)

class GrpcClient(GrpcBase):
    def __init__(self, args, logger:logging.Logger):
        super().__init__()
        self.args = args
        self.logger = logger
        self.server_uri = f'{self.args.master_addr}:{self.args.master_port}'
        self.training_sets = fedscale_dataset.get_train_datasets(logger)
        
        self.lock = asyncio.Lock()
        self.simulation_event = asyncio.Event()
        # 重连
        self.stop = False
        self.retry_delay = 1  # 初始重试延迟（秒）
        self.retry_backoff = 2  # 指数回退因子
        self.max_retries = -1  # 最大重试次数，-1 表示无限重试
        self.retry_count = 0
        
        self.training_task = None
        self.training_active = False
        self.pwd_round = -1
        self.pwd_rank = -1
        
        self.client_id = args.rank
        self.round_statistical_message = {
            "rank": 0,
            "statistical_utility": 0,
            "rate": 0,
            "distribution_time": 0,
            "distribution_start_time": 0,
            "distribution_stop_time": 0,
            "calculation_time": 0,
            "calculation_start_time": 0,
            "calculation_stop_time": 0,
            "aggregation_time": 0,
            "aggregation_start_time": 0,
            "aggregation_stop_time": 0,
            "contribution": 0,
        }

        options = [
            ('grpc.max_send_message_length', 500 * 1024 * 1024),
            ('grpc.max_receive_message_length', 500 * 1024 * 1024),
        ]
        self.logger.info(self.server_uri)
        # 还没通信
        self.channel = grpc.aio.insecure_channel(self.server_uri, options)
        self.stub = grpc_pb2_grpc.BroadcastServiceStub(self.channel)
    
        self.init_params(self.args)
    
    async def reset_statistical_message(self):
        async with self.lock:
            self.round_statistical_message = {
                "rank": 0,
                "statistical_utility": 0,
                "rate": 0,
                "distribution_time": 0,
                "distribution_start_time": 0,
                "distribution_stop_time": 0,
                "calculation_time": 0,
                "calculation_start_time": 0,
                "calculation_stop_time": 0,
                "aggregation_time": 0,
                "aggregation_start_time": 0,
                "aggregation_stop_time": 0,
                "contribution": 0,
            }
    
    # async def prepare_for_training(self):
    #     # 从server接收client的config
    #     config_message = grpc_pb2.ServerMessage(
    #         config_distribute=grpc_pb2.ConfigDistribute(
    #             config=GrpcBase._serialize_model(config_dict),
    #             timestamp=self._get_current_timestamp(),
    #         )
    #     )
    #     await self._broadcast_message(config_message, self.connected_clients.keys())
    #     self.config = utils.to_namespace(config_dict)
    #     self.logger.info(f"{self.config}")

    async def communicate(self):
        """
        连接到服务器并开始通信。
        包含连接失败时的重试逻辑。
        """
        delay = self.retry_delay

        while not self.stop:
            # 尝试建立连接
            while True:
                try:
                    login_message = grpc_pb2.ClientMessage(
                        client_login=grpc_pb2.ClientLogin(
                            rank=self.args.rank,
                            timestamp=self._get_current_timestamp(),
                        )
                    )
                    self.stream = self.stub.Communicate()
                    await self.stream.write(login_message)
                except grpc.aio.AioRpcError:
                    self.logger.exception(f"无法连接到服务器 {self.server_uri}。重试中...")
                    await asyncio.sleep(3)
                else:
                    self.logger.info(f"连接到服务器")
                    break
            
            try:
                self.retry_count = 0  # 重置重试计数
                delay = self.retry_delay  # 重置延迟
                
                try:
                    async for server_message in self.stream:
                        recv_datetime = self._get_current_datetime()
                        await self.handle_message(server_message, recv_datetime)
                except grpc.aio.AioRpcError as e:
                    self.logger.exception(f"RPC 错误: {e}")
                    raise e  # 引发异常以触发外层的重连逻辑
                except asyncio.CancelledError as e:
                    self.logger.exception("Communication task was cancelled.")
                    raise e
                except Exception as e:
                    self.logger.exception(f"An unexpected error occurred during communication: {e}")
                    raise e

            except (grpc.aio.AioRpcError, asyncio.CancelledError) as e:
                self.connected = False
                if self.stop:
                    self.logger.info("Client is stopping. Exiting communicate loop.")
                    break
                self.retry_count += 1
                if self.max_retries != -1 and self.retry_count > self.max_retries:
                    self.logger.error(f"Exceeded maximum retries ({self.max_retries}). Exiting.")
                    break
                self.logger.exception(f"Connection failed: {e}. Retrying in {delay} seconds...")
                await asyncio.sleep(delay)
                delay *= self.retry_backoff  # 指数回退
            except Exception as e:
                self.connected = False
                self.logger.exception(f"An unexpected error occurred: {e}. Retrying in {delay} seconds...")
                self.retry_count += 1
                if self.max_retries != -1 and self.retry_count > self.max_retries:
                    self.logger.exception(f"Exceeded maximum retries ({self.max_retries}). Exiting.")
                    break
                await asyncio.sleep(delay)
                delay *= self.retry_backoff  # 指数回退
            finally:
                await self.close_connection()

    async def handle_message(self, server_message, recv_datetime):
        if server_message.HasField(MessageType.ConfigDistribute.value):
            message = server_message.config_distribute
            config_dict = GrpcBase._deserialize(message.config)
            self.config = utils.to_namespace(config_dict)
            self.logger.info(f"客户端 {self.client_id} 收到配置消息: {self.config}")
            
            self.model_root_dir = self.config.model.ModelPath
            os.makedirs(self.model_root_dir, exist_ok=True)
        elif server_message.HasField(MessageType.ClientProfileDistribute.value):
            message = server_message.client_profile_distribute
            self.logger.info(f"participants_number {self.participants_number}")
            if self.args.running_mode == utils.SIMULATION_MODE:
                self.client_profiles = GrpcBase._deserialize(message.client_profiles)
                self.logger.info(f"client_profiles {len(self.client_profiles)}")
            
            if self.args.transfer_mode == utils.BTPS_TRANSFER_MODE:
                self.torrent = None
                self.path = None
                self.tc = TorrentCommunicationGRPC(self.args.rank, self.logger)
        elif server_message.HasField(MessageType.ClientRankDistribute.value):
            message = server_message.client_rank_distribute
            self.logger.info(f"客户端 {self.client_id} 收到客户端模拟rank消息: {message.simulated_ranks}")
            # 正在执行过时的训练任务
            if self.training_active:
                if self.training_task and not self.training_task.done():
                    self.training_task.cancel()
                    try:
                        await self.training_task
                    except asyncio.CancelledError:
                        self.logger.info(f"Training task {self.client_id} cancelled. #{message.round_number}")
                    except Exception as e:
                        self.logger.info(f"Error: {e}")
                    self.training_active = False
                    await self.reset_statistical_message()
            # 限速
            self.pwd_rank = message.simulated_ranks[self.args.rank]
            RANK = self.args.rank
            self.logger.info(f"[{self.pwd_round}] {RANK} simulate {self.pwd_rank}, {self.client_profiles[self.pwd_rank]}")
            if self.args.running_mode == utils.SIMULATION_MODE:
                if self.args.rate_limit:
                    utils.rate_limit(RANK, self.client_profiles[self.pwd_rank], self.logger)
        elif server_message.HasField(MessageType.TorrentDistribute.value) or server_message.HasField(MessageType.ModelDistribute.value): # 模型分发
            if server_message.HasField(MessageType.TorrentDistribute.value):
                message = server_message.torrent_distribute
            elif server_message.HasField(MessageType.ModelDistribute.value):
                message = server_message.model_distribute
                
            if message.round_number < self.pwd_round:
                self.logger.info(f"客户端 {self.client_id} 收到过时的模型分发消息: {message.round_number} {self.pwd_round}。")
                return
            elif message.round_number == self.pwd_round:
                self.logger.info(f"客户端 {self.client_id} 收到重复的模型分发消息: {message.round_number} {self.pwd_round}。")
                return
            else:
                # 正在执行过时的训练任务
                if self.training_active:
                    if self.training_task and not self.training_task.done():
                        self.training_task.cancel()
                        try:
                            await self.training_task
                        except asyncio.CancelledError:
                            self.logger.info(f"Training task {self.client_id} cancelled. #{message.round_number}")
                        except Exception as e:
                            self.logger.info(f"Error: {e}")
                        self.training_active = False
                        await self.reset_statistical_message()
            
            self.pwd_round = message.round_number
            
            if server_message.HasField(MessageType.TorrentDistribute.value):
                self.logger.info(f"torrent {len(message.torrent)}")
                path = os.path.join(self.model_root_dir,  f"{self.pwd_round}.pth")
                self.logger.debug(path)
                # grpc传输过程中会使用UTC，因此需要手动转换
                distribution_start_time = self._grpc_timestamp_to_datetime(message.timestamp)
                distribution_start_time = self._datetime_add_tzinfo(distribution_start_time)
                self.torrent = message.torrent
                downloading_output = self.tc.bt_recv(self.torrent)
                self.logger.info(f"downloading_output {downloading_output}")
                distribution_stop_time = self._get_current_datetime()
                distribution_time = (distribution_stop_time - distribution_start_time).total_seconds()
                state_dict_bytes = open(path, 'rb').read()
                # contribution = downloading_output['byteswrittendata']
            elif server_message.HasField(MessageType.ModelDistribute.value):
                # grpc传输过程中会使用UTC，因此需要手动转换
                distribution_start_time = self._grpc_timestamp_to_datetime(message.timestamp)
                distribution_start_time = self._datetime_add_tzinfo(distribution_start_time)
                distribution_stop_time = recv_datetime
                distribution_time = (distribution_stop_time - distribution_start_time).total_seconds()
                state_dict_bytes = message.state_dict
                # contribution = 0
            
            async with self.lock:
                self.round_statistical_message['distribution_time'] = distribution_time
                self.round_statistical_message['distribution_start_time'] = distribution_start_time
                self.round_statistical_message['distribution_stop_time'] = distribution_stop_time
                # self.round_statistical_message['contribution'] = contribution
                
            self.logger.info(f"finish distribution: {distribution_time}, {distribution_start_time} {distribution_stop_time}")
            self.logger.info(f"rate {len(state_dict_bytes)*8/1024/1024/distribution_time} Mb/s")
            self.logger.info(f"客户端 {self.client_id} 收到服务器数据并成功加载模型。")

            # self.training_done = Event()
            # self.training_task = asyncio.create_task(self.train(message, recv_datetime, training_done))
            self.training_task = asyncio.create_task(self.train(state_dict_bytes))
            # self.send_model_udpates_task = asyncio.create_task(self.send_model_updates(training_done, message))
            self.send_model_udpates_task = asyncio.create_task(self.send_model_updates(self.training_task, message))
            # self.training_task = asyncio.create_task(self.train_and_send(message, recv_datetime))
        elif server_message.HasField(MessageType.SimulationMessage.value):
            message = server_message.simulation_message
            self.simulation_message = message
            # check rank and client id
            if message.client_id != self.client_id or message.rank != self.pwd_rank:
                raise ValueError
            self.simulation_event.set()
        elif server_message.HasField(MessageType.RoundTerminate.value):
            message = server_message.round_terminate
            if self.training_task and not self.training_task.done():
                self.training_task.cancel()
                try:
                    await self.training_task
                except asyncio.CancelledError:
                    self.logger.info("Training task cancelled.")
                except Exception as e:
                    self.logger.info(f"Error: {e}")
                self.training_active = False
                await self.reset_statistical_message()
                self.logger.info(f"客户端 {self.client_id} 收到轮次终止消息: #{message.round_number}")
        elif server_message.HasField(MessageType.TrainingTerminate.value):
            message = server_message.training_terminate
            if self.training_task and not self.training_task.done():
                self.training_task.cancel()
                try:
                    await self.training_task
                except asyncio.CancelledError:
                    self.logger.info("Training task cancelled.")
            self.logger.info(f"客户端 {self.client_id} 收到训练终止消息")
            self.stop = True
            await self.close_connection()
            return  # 正常终止通信

    async def close_connection(self):
        """关闭连接"""
        if hasattr(self, 'stream') and self.stream:
            await self.stream.done_writing()
            self.stream.cancel()
            # await self.channel.close()
        self.connected = False
        self.logger.info(f"客户端 {self.client_id} 连接已关闭。")

    # async def send_model_updates(self, training_done:asyncio.Event, message):
    async def send_model_updates(self, training_task:asyncio.Future, message):
        """
            发送模型更新到服务器。
        """
        # 等待训练的完成
        # 如果训练被终止，则取消发送
        # 如果训练完成了，且没有被终止，则发送模型更新
        pwd_rank = self.pwd_rank
        
        try:
            await training_task # 等待训练完成
            self.logger.info("wait over")
        except asyncio.CancelledError: # 训练被RoundTerminate终止
            self.logger.info("Training was cancelled. Aborting model updates.")
        else:
            # 训练完成且没有被终止
            # 发送消息，不可终止
            try:
                # 将接收到的数据发送回服务器
                aggregatime_start_time = self._get_current_datetime()
                client_message = grpc_pb2.ClientMessage(
                    model_aggregate=grpc_pb2.ModelAggregate(
                        state_dict=self.training_result["output_state_dict"],
                        timestamp=self._get_current_timestamp(),
                        rank=pwd_rank,
                        dataset_length=self.training_result['sample_number'],
                        round_number=message.round_number,
                    )
                )
                await self.stream.write(client_message)
                aggregation_stop_time = self._get_current_datetime()
                aggregatime_time = (aggregation_stop_time - aggregatime_start_time).total_seconds()
                self.logger.info(f"客户端 {self.client_id} 已发送数据回服务器。")
                
                async with self.lock:
                    if self.args.transfer_mode == utils.BTPS_TRANSFER_MODE:
                        output = self.tc.stop_seeding(self.torrent)
                        self.round_statistical_message['contribution'] = output['byteswrittendata']
                        self.logger.info(f"byteswrittendata {self.round_statistical_message['contribution']}")
                    
                    self.round_statistical_message['aggregation_time'] = aggregatime_time
                    self.round_statistical_message['aggregation_start_time'] = aggregatime_start_time
                    self.round_statistical_message['aggregation_stop_time'] = aggregation_stop_time
                
                    selection_message = grpc_pb2.ClientMessage(
                        selection_message=grpc_pb2.SelectionMessage(
                            data=GrpcBase._serialize(self.round_statistical_message),
                            timestamp=self._get_current_timestamp(),
                        )
                    )
                    await self.stream.write(selection_message)
                    self.logger.info(f"客户端 {self.client_id} 已发送统计信息。")
                
                
            except asyncio.CancelledError:
                self.logger.info("训练任务被取消。")
                # raise  # 重新引发以允许外部处理取消
            except AioRpcError as e:
                self.logger.exception(f"gRPC错误: {e}")
                # 根据需要处理特定的 gRPC 错误
            except Exception as e:
                self.logger.exception(f"返回数据错误: {e}")
            finally:
                self.training_active = False
                self.logger.info("训练任务已结束。")

    async def train(self, state_dict_bytes):
        """
            训练模型并将更新发送回服务器。
        """
        self.training_active = True
        pwd_rank = self.pwd_rank

        # train
        train_dataloader = fedscale_dataset.select_dataset(
            pwd_rank, self.training_sets,
            batch_size=parser.args.batch_size, args=parser.args,
            collate_fn=fedscale_dataset.get_collate_fn()
        )
        # device = torch.device(f"cuda:{self.args.rank%torch.cuda.device_count()}")
        device = None
        if self.args.rate_limit:
            utils.rate_limit(self.args.rank, self.client_profiles[self.pwd_rank], self.logger, limit=False)
            ret = request_computation_node(self.args, state_dict_bytes, pwd_rank, self.logger, train_dataloader, device)
            utils.rate_limit(self.args.rank, self.client_profiles[self.pwd_rank], self.logger)
        else:
            ret = request_computation_node(self.args, state_dict_bytes, pwd_rank, self.logger, train_dataloader, device)
            
        self.logger.info(f"{ret['statistical_utility']}, {ret['sample_number']}, {ret['calculation_start_time']}, {ret['calculation_stop_time']}, {ret['calculation_time']}")
        
        async with self.lock:
            self.round_statistical_message['rank'] = pwd_rank
            self.round_statistical_message['statistical_utility'] = ret['statistical_utility']
            self.round_statistical_message['rate'] = self.client_profiles[pwd_rank]['communication']

        self.logger.info(f"{self.round_statistical_message}")
        
        self.training_result = ret
        
        # 修正计算时间
        if self.args.running_mode == utils.SIMULATION_MODE:
            calculation_time = 3 * self.args.local_epoch * ret['sample_number'] * self.client_profiles[pwd_rank]['computation'] / 1000.0 / 10
            if not self.args.quick_simulate:
                try:
                    # 发送 time(distribution_time, calculation_time)
                    data = {
                        "distribution_time": self.round_statistical_message['distribution_time'],
                        "calculation_time": calculation_time
                    }
                    simulation_message = grpc_pb2.ClientMessage(
                        simulation_message=grpc_pb2.SimulationMessage(
                            data=GrpcBase._serialize(data),
                            rank=pwd_rank,
                            client_id=self.client_id,
                        )
                    )
                    await self.stream.write(simulation_message)
                except Exception as e:
                    self.logger.exception(f"发送模拟信息失败: {e}")
                else:
                    self.logger.info(f"客户端 {self.client_id} 已发送模拟信息。")

                # 服务器端修正计算时间
                self.simulation_event.clear()
                await self.simulation_event.wait() # barrier
                # 等待时间 = time - min(time)
                simulation_sleep_time = GrpcBase._deserialize(self.simulation_message.data)['sleep_time']
                self.logger.info(f"客户端 {self.client_id} 收到的模拟休眠信息: {simulation_sleep_time}")
                await asyncio.sleep(simulation_sleep_time)

            calculation_start_time = self._get_current_datetime()
            calculation_stop_time = GrpcBase._add_seconds(calculation_start_time, calculation_time)

            # 完成后开始聚合
            async with self.lock:
                self.round_statistical_message['calculation_time'] = calculation_time
                self.round_statistical_message['calculation_start_time'] = calculation_start_time
                self.round_statistical_message['calculation_stop_time'] = calculation_stop_time
        else:
            async with self.lock:
                self.round_statistical_message['calculation_time'] = ret['calculation_time']
                self.round_statistical_message['calculation_start_time'] = self._time_to_datetime(ret['calculation_start_time']) 
                self.round_statistical_message['calculation_stop_time'] = self._time_to_datetime(ret['calculation_stop_time'])
        
        self.logger.info("wait over")
        
        # traing_done.set() # 训练完成
        
        # try:
        #     # 将接收到的数据发送回服务器
        #     client_message = grpc_pb2.ClientMessage(
        #         model_aggregate=grpc_pb2.ModelAggregate(
        #             state_dict=ret["output_state_dict"],
        #             timestamp=self._get_current_timestamp(),
        #             rank=pwd_rank,
        #             dataset_length=ret['sample_number'],
        #             round_number=message.round_number,
        #         )
        #     )
        #     await self.stream.write(client_message)
        #     self.logger.info(f"客户端 {self.client_id} 已发送数据回服务器。")
        # exacept asyncio.CancelledError:
        #     self.logger.info("训练任务被取消。")
        #     # raise  # 重新引发以允许外部处理取消
        # except AioRpcError as e:
        #     self.logger.exception(f"gRPC错误: {e}")
        #     # 根据需要处理特定的 gRPC 错误
        # except Exception as e:
        #     self.logger.exception(f"返回数据错误: {e}")
        # finally:
        #     self.training_active = False
        #     self.logger.info("训练任务已结束。")

def ping(logger):
    StackName = os.getenv("StackName")
    hostname = os.getenv("Hostname")
    computaion_node_domain_name = f"{StackName}_computation_node_{hostname}"

    while True:
        try:
            response = requests.get(f"http://{computaion_node_domain_name}:27500/ping", timeout=120)
        except Exception as e:
            logger.warning(e)
        else:
            logger.info(f"ping ok {response}")
            break

def request_computation_node(args, state_dict_bytes, pwd_rank, logger, train_dataloader, device=None):
    if device is not None:
        model, debug_param, statistical_utility, sample_number, calculation_start_time, calculation_stop_time, calculation_time = train_handler(args, logger, GrpcBase._deserialize(state_dict_bytes), train_dataloader, device)
        model = model.to("cpu")
        result = {
            "output_state_dict": GrpcBase._serialize(model.state_dict()),
            "statistical_utility":statistical_utility,
            "sample_number":sample_number,
            "calculation_start_time":calculation_start_time,
            "calculation_stop_time":calculation_stop_time,
            "calculation_time":calculation_time,
        }
    else:
        # {StackName}_computation_node_{id} id from 0 to the number of physical machines
        StackName = os.getenv("StackName")
        hostname = os.getenv("Hostname")
        # PhysicalClientMachineNumber = int(os.getenv("PhysicalClientMachineNumber"))
        # id = random.choice(list(range(PhysicalClientMachineNumber)))
        # {StackName}_computation_node_{id} id from 0 to the number of physical machines
        computaion_node_domain_name = f"{StackName}_computation_node_{hostname}"

        # bytes 转base64 str
        s = base64.b64encode(state_dict_bytes)
        s += b'=' * (-len(s) % 4)
        encoded_state_dict = s.decode('utf-8')
        params = {
            "args": vars(args),
            "input_state_dict": encoded_state_dict,
            "pwd_rank": pwd_rank,
        }
        ping(logger)
        while True:
            try:
                response = requests.post(f"http://{computaion_node_domain_name}:27500/compute", json=params, timeout=120)
            except (RemoteDisconnected, ProtocolError, ConnectionError, ReadTimeout, JSONDecodeError) as e:
                logger.warning(e)
                time.sleep(3)
            except Exception as e:
                logger.warning(e)
                exit(0)
            else:
                if response.status_code == 200:
                    result = response.json()
                    break
                else:
                    logger.warning(f"request computation node error: {response.text}")
                    time.sleep(3)

        result['output_state_dict'] = base64.b64decode(utils.padding_base64_str(result['output_state_dict']))
    
    return result

def train_handler(args, logger, state_dict, train_dataloader, device):
    """
        return model, debug_param, total_loss, sample_number, calculation_start_time, calculation_stop_time, calculation_time
    """
    # init model
    # model = models.get_model(args.model, args.dataset, args.num_classes)
    model = fedscale_dataset.init_model()
    model.load_state_dict(state_dict)
    model = model.to(device)
    if args.gradient_policy == utils.AggregateType.FEDPROX_STRATEGY:
        # global_model = [param.data.clone() for param in model.parameters()]
        global_model = deepcopy(model)
        # fedprox_optimizer = client_optimizer.ClientOptimizer()
    # set reduction to none to keep each sample's loss
    loss_func = nn.CrossEntropyLoss(reduction='none')
    # loss_func = nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    # optimizer = torch.optim.Adam(model.parameters())
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    # # evaluation
    # result = evaluate(deepcopy(model), device, epoch, 0, test_dataloader, nn.CrossEntropyLoss())
    # logger.info(f"[epoch: {epoch}] acc: {result[0]} loss: {result[1]}")

    # calculation
    debug_param = next(model.parameters())
    logger.debug(f"before train, param:\n{debug_param[0][0]}")
    model.train()
    # create an empty tensor to store all samples' loss
    total_loss = torch.Tensor().to(device)
    model.train()
    calculation_start_time = time.time()

    for local_epoch in range(args.local_epoch):
        sample_number = 0
        for pwd_iter, (x, y) in enumerate(train_dataloader):
            logger.info(f"{local_epoch} {pwd_iter}")
            # Here, clients are computing.
            pwd_iter += 1
            sample_number += len(y)
            optimizer.zero_grad()

            if args.dataset == "google_speech_commands":
                x = torch.unsqueeze(x, 1)

            x, y = x.to(device), y.to(device)

            y_hat = model(x)
            # if args.gradient_policy == utils.AggregateType.FEDPROX_STRATEGY:
            #     proximal_term = 0.0
            #     for w, w_t in zip(model.parameters(), global_model.parameters()):
            #         proximal_term += (w - w_t).norm(2)

            losses = loss_func(y_hat, y)
            # if args.gradient_policy != utils.AggregateType.FEDPROX_STRATEGY:
            #     loss = losses.mean()
            # else:
            #     loss = losses.mean() + (args.proxy_mu / 2) * proximal_term

            # loss.backward()
            losses.mean().backward()
            for param in model.parameters():
                if param.requires_grad:
                    # logger.debug(f"param.grad[0][0]: {param.grad[0][0]}")
                    break
            optimizer.step()
            total_loss = torch.cat((total_loss, losses.detach().clone()))

            # fedprox
            # if args.gradient_policy == utils.AggregateType.FEDPROX_STRATEGY:
            #     fedprox_optimizer.update_client_weight(args, model, global_model)

    calculation_stop_time = time.time()
    calculation_time = calculation_stop_time - calculation_start_time
    logger.info(f"actual calculation time: {timestamp_to_datetime(calculation_stop_time)} - {timestamp_to_datetime(calculation_start_time)} = {calculation_time}")
    statistical_utility = utils.calculate_statistical_utility(total_loss, sample_number)
    torch.cuda.empty_cache()
    
    return model, debug_param, statistical_utility, sample_number, calculation_start_time, calculation_stop_time, calculation_time



async def run_client(client_id):
    client = GrpcClient(client_id)
    await client.communicate()

if __name__ == '__main__':
    import sys
    if len(sys.argv) != 2:
        print("用法: python client.py <client_id>")
    else:
        asyncio.run(run_client(sys.argv[1]))