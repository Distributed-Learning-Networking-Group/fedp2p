import time
import torch
import io
import pytz
import threading
import queue
import os

from datetime import datetime
from datetime import timezone
from datetime import timedelta
from google.protobuf.timestamp_pb2 import Timestamp
from enum import auto, Enum
from d2l import torch as d2l

class GrpcBase:
    def __init__(self):
        self.beijing_tz = pytz.timezone('Asia/Shanghai')
    
    def _time_to_datetime(self, t: float):
        return datetime.fromtimestamp(t, self.beijing_tz)
    
    def _get_current_datetime(self):
        return datetime.now(self.beijing_tz)
        
    def _get_current_timestamp(self):
        """返回指定时区的 Timestamp 对象"""
        now = self._get_current_datetime()
        timestamp = Timestamp()
        timestamp.FromDatetime(now)
        return timestamp

    def _grpc_timestamp_to_datetime(self, timestamp:Timestamp)->datetime:
        """将 GRPC 传输的 Timestamp 对象(UTC) 转换为指定时区的 datetime 对象"""
        return timestamp.ToDatetime().replace(tzinfo=timezone.utc).astimezone(self.beijing_tz)
    
    def _datetime_add_tzinfo(self, dt:datetime)->datetime:
        """将 datetime 对象添加时区信息"""
        return dt.astimezone(self.beijing_tz)

    def format_datetime(self, dt:datetime):
        """将 datetime 对象转换为可读的字符串格式"""
        return dt.strftime('%Y-%m-%d %H:%M:%S')
    
    @staticmethod
    def _serialize(data):
        """将对象序列化为字节"""
        buffer = io.BytesIO()
        torch.save(data, buffer)
        return buffer.getvalue()

    @staticmethod
    def _deserialize(data):
        """将字节反序列化为对象"""
        buffer = io.BytesIO(data)
        return torch.load(buffer)
    
    @staticmethod
    def _add_seconds(dt:datetime, seconds:float):
        return dt + timedelta(seconds=seconds)
    
    def init_params(self, args):
        if args.dataset == "femnist":
            self.participants_number = 2800
        elif args.dataset == "google_speech_commands":
            self.participants_number = 2167
        elif args.dataset == "openimage":
            self.participants_number = 11325
        else:
            raise ValueError
        self.logger.info(f"participants_number {self.participants_number}")

        # if args.use_gpu:
        #     self.device = d2l.try_gpu()
        # else:
        #     self.device = torch.device("cpu")

        if args.use_gpu and torch.cuda.is_available():
            device = os.getenv("device")
            if device is not None: # 选择指定的设备
                self.device = torch.device(device)
            else: # 从可选设备中进行尝试
                self.device = d2l.try_gpu()
        else: # 不使用GPU或GPU不可用
            self.device = torch.device("cpu")
        self.logger.info(f"device: {self.device}")
        
# 常量定义
class MessageType(Enum):
    ConfigDistribute = "config_distribute" # server -> client
    ClientProfileDistribute = "client_profile_distribute" # server -> client
    ClientRankDistribute = "client_rank_distribute" # server -> client
    ModelDistribute = "model_distribute" # server -> client
    TorrentDistribute = "torrent_distribute" # server -> client
    RoundTerminate = "round_terminate" # server -> client: terminate this round of training 
    TrainingTerminate = "training_terminate" # server -> client: terminate the whole training process

    ModelAggregate = "model_aggregate" # client -> server
    ClientLogin = "client_login" # client -> server
    SelectionMessage = "selection_message" # client -> server
    SimulationMessage = "simulation_message" # client -> server

WS_MAX_SIZE = 2**30


if __name__ == "__main__":
    base = GrpcBase()
    print(base._get_current_timestamp())
    print(base._serialize("hello"))
    print(base._parse_timestamp(base._get_current_timestamp()))
    print(base._deserialize(base._serialize("hello")))