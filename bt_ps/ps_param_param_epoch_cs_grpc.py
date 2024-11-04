import sys
sys.path.append("/app/bt_ps")
sys.path.append("/app/bt_ps/thirdparty/FedScale")
print(sys.path)
import os
import time
import torch
import random
import requests
import pickle
import threading
import queue
import asyncio
import logging
import torch.nn as nn

from torch import distributed as dist
from d2l import torch as d2l
from torch.utils.tensorboard import SummaryWriter
from requests.exceptions import ConnectionError, ReadTimeout
from urllib3.exceptions import ProtocolError
from json.decoder import JSONDecodeError
from queue import Empty
from http.client import RemoteDisconnected
from copy import deepcopy

import grpc_server
import grpc_client

from p2p_server.models import models
from p2p_server.rpc.rpc_client import TorrentCommunicationPyTorch, TorrentCommunicationGRPC
from p2p_server.utils import utils
from client_manager import ClientManager
from fedscale.utils import dataset as fedscale_dataset


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
            logger.info(f"evaluate params: {task}")
            result = evaluate(*task)
            result_queue.put((result, task[2]))


def transfer_client_selection_result(selected_clients:torch.Tensor=None):
    """
        selected_clients[index] = 1 means that the client with index is selected.

        selected_clients[index] = 0 means that the client with index is not selected.

        selected_clients[0] is always 1, which means that the server is always selected.

        For server, selected_clients is not None. Server will broadcast the selection result to all clients.

        For clients, selected_clients is None.

        return: selected_clients, sub_group

        Args:
            selected_clients (list, optional): For server, it's not None. For clients, it's None. Defaults to None.

        Raises:
            ValueError: raised when selected_clients is None when this function is called by server.

        Returns:
            selected_clients_rank: a list that contains server's and selected clients' rank.
            sub_group: torch's process group of the selected clients.
    """
    RANK, WORLD_SIZE = dist.get_rank(), dist.get_world_size()

    if RANK == 0:
        if selected_clients is None:
            raise ValueError("selected_clients should not be None when this function is called by server.")
        selected_clients_rank = torch.nonzero(selected_clients).flatten()
        scatter_list = [torch.unsqueeze(i, 0) for i in selected_clients_rank]
    else:
        scatter_list = None
    pwd_rank = torch.empty(1, dtype=torch.int64)
    try:
        dist.scatter(pwd_rank, scatter_list, src=0)
    except Exception as e:
        print(e, type(e))
    return pwd_rank.item()


def scatter_python_objects(object_list=None, subgroup=None):
    """
        For receiver, object_list/scatter_object_input_list is None.

        For sender, object_list/scatter_object_input_list is not None.

        For both, it's enough for output to be [None].
    """
    output = [None]
    dist.scatter_object_list(output, object_list, 0, subgroup)
    return output[0]


async def run_server(args, logger):
    listen_addr = f"{args.master_addr}:{args.master_port}"
    server, service = grpc_server.create_server(args, logger)
    await server.start()
    logger.info(f"gRPC 服务器已启动，监听 {listen_addr}")
    
    # 进行训练之前的准备
    await service.prepare_for_training()
    logger.info("gRPC 服务器已完成训练前的准备及工作")
    # 启动训练进程
    asyncio.create_task(service.start_new_training_round()) # only one is executed
    
    # 等待服务器关闭事件
    await service.shutdown_event.wait()

    # 关闭服务器
    await server.stop(5)
    logger.info("gRPC 服务器已关闭。")

def server_func(args, logger):
    RANK, WORLD_SIZE = dist.get_rank(), dist.get_world_size()
    config = utils.update_config_file(RANK, args.master_addr, args.master_port, args.model, args.dataset, logger)
    model_root_dir = config.model.ModelPath
    os.makedirs(model_root_dir, exist_ok=True)

    # init model
    # model = models.get_model(args.model, args.dataset, args.num_classes)
    # 注意修改yml而非命令行参数
    model = fedscale_dataset.load_model()
    # load from checkpoint
    checkpoint_model_name = f"{args.model}_{args.dataset}_backup.pth"
    log_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    log_path = os.path.join(log_path, "logs", os.getenv("DateTime"))
    if os.path.exists(checkpoint_model_name):
        model.load_state_dict(torch.load(checkpoint_model_name))
        model_size = os.path.getsize(checkpoint_model_name)
    else:
        torch.save(model.state_dict(), os.path.join(log_path, checkpoint_model_name))
        model_size = os.path.getsize(os.path.join(log_path, checkpoint_model_name))
    logger.info(f"model_size {model_size}")

    debug_param = next(model.parameters())

    if args.use_gpu:
        device = d2l.try_gpu()
    else:
        device = torch.device("cpu")

    if args.dataset == "femnist":
        participants_number = 2800
    elif args.dataset == "google_speech_commands":
        participants_number = 2167
    elif args.dataset == "openimage":
        participants_number = 11325
    else:
        raise ValueError
    logger.info(f"participants_number {participants_number}")

    # In simulation mode, server sends each client's computation and communication capacity.
    if args.running_mode == utils.SIMULATION_MODE:
        profiles = utils.load_client_profile()
        # upload rate should range from 10 to 50 mbps
        feasible_profiles = []
        for profile in profiles.values():
            upload_rate = profile['communication'] / 1024
            download_rate = upload_rate
            if download_rate >= 20:
                feasible_profiles.append(profile)

        client_profiles = []
        for index in random.sample(range(1, len(feasible_profiles)), 1+participants_number):
            client_profiles.append(feasible_profiles[index])
        # if args.model == "resnet18" and args.dataset == "femnist":
        #     for profile in client_profiles:
        #         profile['computation'] /= 4
        # elif args.model == "resnet34" and args.dataset == "google_speech_commands":
        #     for profile in client_profiles:
        #         profile['computation'] /= 3
        logger.info(f"client_profiles {client_profiles} {len(client_profiles)}")
        dist.broadcast_object_list(client_profiles, 0)
        # client_profile = scatter_python_objects(client_profiles)

        # set server profile
        # 1000*1024kbit/s
        server_profile = client_profiles[0]
        if args.model == "resnet18" and args.dataset == "femnist":
            server_profile['communication'] = 500*1024 # kbps
        elif args.model == "resnet34" and args.dataset == "google_speech_commands":
            server_profile['communication'] = 1000*1024 # kbps
        elif args.model == "shufflenet_v2_x2_0" and args.dataset == "openimage":
            server_profile['communication'] = 300*1024 # kbps
        else:
            server_profile['communication'] = 1000*1024 # kbps
            
        server_profile['computation'] = 0
        logger.info(f"server_profile {server_profile}")

        if args.rate_limit:
            utils.rate_limit(RANK, server_profile, logger)
    
    if args.transfer_mode == utils.BTPS_TRANSFER_MODE:
        torrent = None
        path = None
        tc = TorrentCommunicationPyTorch(logger)

    # dataset
    # clients_data, test_dataloader = fedscale_dataset.load_dataset(client_num=WORLD_SIZE-1)
    test_dataloader = fedscale_dataset.get_test_dataloader(logger)

    # # 传输index
    # clients_data.insert(0, [0])
    # scatter_python_objects(clients_data)
    # size = dict()
    # for index, client_data in enumerate(clients_data):
    #     size[f'{index}'] = len(client_data)
    # logger.info(f"clients' data size: {size}")

    # train_dataset, train_dataloader, test_dataset, test_dataloader = datasets.load_dataset(
    #     args.dataset, args.data_dir, args.batch_size, args
    # )

    # train
    # iter_num = math.ceil(len(train_dataset)/(args.world_size-1)/args.batch_size)
    loss_func = nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-3)
    # optimizer = torch.optim.Adam(model.parameters())

    checkpoint_cm_name = f"{args.model}_{args.dataset}_cm.backup"
    client_manager = ClientManager(args, logger, participants_number, model_size)
    if os.path.exists(checkpoint_cm_name):
        with open(checkpoint_cm_name, "rb") as f:
            client_manager = pickle.load(f)
            print(client_manager.participants_number)
    
    training_sets = fedscale_dataset.get_train_datasets(logger)
    client_manager.init_client_info(training_sets)

    total_indeed_epoch_time = 0
    total_simulated_epoch_time= 0
    selection_messages = None
    summary_writer = SummaryWriter(
        os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
            "logs",
            os.getenv("DateTime")
        )
    )

    evaluate_task_queue = queue.Queue(maxsize=1)
    evaluate_result_queue = queue.Queue(maxsize=1)
    evaluate_thread = threading.Thread(
        target=evaluate_handler, 
        args=(
            evaluate_task_queue, 
            evaluate_result_queue,
            logger
        )
    )
    evaluate_thread.start()
    logger.info("start evaluate thread")

    distribution_time_list = []
    average_local_distribution_time_list = []

    calculation_time_list = []
    averate_local_calculation_time_list = []

    aggregation_time_list = []
    average_local_aggregation_time_list = []

    for epoch in range(1, args.num_epochs+1):
        dist.barrier()
        # 记录一个epoch的时间
        pwd_epoch_start_time = time.time()
        aggregation_time = 0
        distribution_time = 0
        
        # client selection
        selected_clients = client_manager.select_clients(epoch, logger, selection_messages)
        selected_clients_rank = torch.nonzero(selected_clients).flatten()
        logger.info(f"[{epoch}] selected_clients_rank: {selected_clients_rank}")

        # 向所有docker传输要模拟的client index/rank
        pwd_rank = transfer_client_selection_result(selected_clients)
        logger.info(f"[{epoch}] {RANK} simulate {pwd_rank}")
        client_total_rate = 0
        for rank in selected_clients_rank[1:]:
            client_total_rate += client_profiles[rank]['communication']
        logger.info(f"upload rate: {client_total_rate/1024}mbps, download rate: {client_total_rate/1024}mbps, server rate: {server_profile['communication']/1024}mbps")

        # distribution
        if args.transfer_mode == utils.BTPS_TRANSFER_MODE:
            if torrent is not None:
                tc.stop_seeding(torrent)
                os.remove(path)
            path = os.path.join(model_root_dir,  f"{epoch}.pth")
            torch.save(model.state_dict(), path)
            
            dist.barrier()
            distribution_start_time = time.time()
            torrent = tc.bt_broadcast(path)
            # stop time should come from clients
        elif args.transfer_mode == utils.PS_TRANSFER_MODE:
            # # params broadcast            
            # # for params in model.parameters():
            # for name, params in model.state_dict().items():
            #     start = time.time()
            #     dist.broadcast(params, 0, group=sub_group)
            #     distribution_time += (time.time()-start)

            # # state_dict broadcast
            # state_dict_tensor = utils.python_object_to_tensor(model.state_dict())
            # state_dict_tensor = utils.broadcast_unfixed_length_tensor(state_dict_tensor, 0, group=sub_group)

            # state_dict send/recv
            dist.barrier()
            distribution_start_time = time.time()
            # size
            state_dict_tensor = utils.python_object_to_tensor(model.state_dict())
            size = torch.tensor([state_dict_tensor.shape[0]], dtype=torch.int64)
            dist.broadcast(size, 0)
            # data
            # send asynchronously
            work_list = []
            # for rank in selected_clients_rank[1:]:
            for rank in range(1, len(selected_clients_rank)):
                work = dist.isend(state_dict_tensor, rank)
                work_list.append(work)
            for work in work_list:
                work.wait()
            # distribution_stop_time = time.time()
            # distribution_time = distribution_stop_time - distribution_start_time
            
        # calculation
        if args.running_mode == utils.SIMULATION_MODE:
            dist.barrier()
        
        # aggregation
        
        # # params broadcast
        # # model.parameters()只能获取可学习的参数，model.state_dict()能获取所有参数，包括BN层的缓存参数
        # # for params in model.parameters():
        # state_dict = dict()
        # for name, params in model.state_dict().items():
        #     # tmp = torch.zeros_like(params.data)
        #     tmp = torch.zeros_like(params, device=torch.device("cpu"))
        #     start = time.time()
        #     # clear params before reduce
        #     dist.reduce(tmp, 0, dist.ReduceOp.SUM, group=sub_group)
        #     aggregation_time += (time.time()-start)
        #     # tmp = (tmp / (args.world_size-1)).to(tmp.dtype)
        #     # tmp = tmp / (args.world_size-1)
        #     tmp = tmp / (len(selected_clients_rank)-1)
        #     state_dict[name] = tmp

        # # state_dict broadcast
        # # size
        # size = utils.python_object_to_tensor(model.state_dict()).size()
        # gather_list = [torch.empty(size, device=torch.device("cpu")) for _ in selected_clients_rank]
        # dist.gather(gather_list[0], gather_list, 0, group=sub_group)
        # gather_list.pop(0)
        # state_dicts = [utils.python_object_to_tensor(state_dict, reverse=True) for state_dict in gather_list]
        # state_dict = dict()
        # for client_state_dict in state_dicts:
        #     for name, params in client_state_dict.items():
        #         if name not in state_dict:
        #             state_dict[name] = params
        #         else:
        #             state_dict[name] += params

        # for name, params in state_dict.items():
        #     state_dict[name] = params / (len(selected_clients_rank)-1)

        # state_dict send/recv
        # TODO: 第一个位置放数据量（int64），后面放state_dict
        state_dict_tensor = utils.python_object_to_tensor(model.state_dict())
        dataset_length_tensor = utils.python_object_to_tensor(str("000000"))
        data_tensor = torch.cat([dataset_length_tensor, state_dict_tensor])
        logger.info(f"state_dict_tensor.size() {state_dict_tensor.size()}, dataset_length_tensor.size() {dataset_length_tensor.size()}, data_tensor.size() {data_tensor.size()}")
        
        # recv asynchronously
        data_tensors = [torch.empty(data_tensor.size(), dtype=torch.uint8, device=torch.device("cpu")) for _ in range(len(selected_clients_rank)-1)]
        work_list = []
        for rank in range(1, len(selected_clients_rank)):
            work = dist.irecv(data_tensors[rank-1], rank)
            work_list.append(work)
        for work in work_list:
            work.wait()
        state_dict = dict()
        # fedavg
        total_length = 0
        for data_tensor in data_tensors:
            dataset_length = int(utils.python_object_to_tensor(data_tensor[:dataset_length_tensor.size()[0]], reverse=True))
            logger.info(f"dataset_length {dataset_length}")
            logger.info(f"dataset_length {dataset_length} {data_tensor[:dataset_length_tensor.size()[0]]} {data_tensor[dataset_length_tensor.size()[0]:20]} {data_tensor[dataset_length_tensor.size()[0]:20].dtype} {data_tensor[:dataset_length_tensor.size()[0]].dtype} {data_tensor.dtype}")

            total_length += dataset_length
            client_state_dict = utils.python_object_to_tensor(data_tensor[dataset_length_tensor.size()[0]:], reverse=True)
            for name, params in client_state_dict.items():
                if name not in state_dict:
                    state_dict[name] = params * dataset_length
                else:
                    state_dict[name] += (params * dataset_length)
        for name, params in state_dict.items():
            # state_dict[name] = params / (len(selected_clients_rank)-1)
            state_dict[name] = params / total_length

        model.load_state_dict(state_dict)
        # backup
        torch.save(model.state_dict(), os.path.join(log_path, checkpoint_model_name))
        with open(os.path.join(log_path, checkpoint_cm_name), "wb") as f:
            pickle.dump(client_manager, f)

        # TODO: transfer messages for selection
        # DONE
        # 1) in simulation mode, selection_messages are required to corrcet the calculation_time by the minimum calculation_time.
        # 2) in oort, selection_messages are required to calculate the statistical utility.
        # size
        size_list = [torch.zeros(1, dtype=torch.int64) for _ in selected_clients_rank]
        dist.gather(
            torch.zeros(1, dtype=torch.int64),
            size_list, 0
        )
        size_list[0] = size_list[1]
        logger.debug(f"size_list: {size_list}")
        selection_messages_tensor = [torch.zeros(size, dtype=torch.uint8) for size in size_list]
        dist.gather(
            torch.zeros(size_list[0], dtype=torch.uint8), 
            selection_messages_tensor, 0
        )
        # remove server's selection message which is an invalid serialization of python object.
        selection_messages_tensor.pop(0)
        selection_messages = [
            utils.python_object_to_tensor(selection_message_tensor, reverse=True) \
            for selection_message_tensor in selection_messages_tensor
        ]
        logger.info(f"selection_messages {selection_messages} shape {selection_messages_tensor[0].shape}")
        
        pwd_epoch_end_time = time.time()
        
        distribution_stop_time_list = []
        local_distribution_time_list = []
        
        calculation_start_time_list = []
        calculation_stop_time_list = []
        local_calculation_time_list = []

        aggregation_start_time_list = []
        aggregation_stop_time_list = []
        local_aggregation_time_list = []
        
        for selection_message in selection_messages:
            # convert str to int
            selection_message['rank'] = int(selection_message['rank'])

            distribution_stop_time_list.append(selection_message['distribution_stop_time'])
            local_distribution_time_list.append(selection_message['distribution_time'])

            calculation_start_time_list.append(selection_message['calculation_start_time'])
            calculation_stop_time_list.append(selection_message['calculation_stop_time'])    
            local_calculation_time_list.append(selection_message['calculation_time'])

            aggregation_start_time_list.append(selection_message['aggregation_start_time'])
            aggregation_stop_time_list.append(selection_message['aggregation_stop_time'])
            local_aggregation_time_list.append(selection_message['aggregation_time'])

        distribution_time = max(distribution_stop_time_list) - distribution_start_time
        average_local_distribution_time = sum(local_distribution_time_list) / len(local_distribution_time_list)
        
        calculation_time = max(calculation_stop_time_list) - min(calculation_start_time_list)
        average_local_calculation_time = sum(local_calculation_time_list) / len(local_calculation_time_list)

        aggregation_time = max(aggregation_stop_time_list) - min(aggregation_start_time_list)
        average_local_aggregation_time = sum(local_aggregation_time_list) / len(local_aggregation_time_list)

        for params in model.parameters():
            logger.debug(f"params.sum(): {params.sum()}")
            break

        indeed_epoch_time = pwd_epoch_end_time-pwd_epoch_start_time
        total_indeed_epoch_time += indeed_epoch_time

        # simulated_epoch_time = max(aggregation_stop_time_list) - distribution_start_time
        simulated_epoch_time = (max(aggregation_stop_time_list) - min(calculation_start_time_list)) + min(local_distribution_time_list)

        # evaluate
        logger.info("evaluate")
        logger.info(f"len(evaluate_task_queue)={evaluate_task_queue.qsize()}")
        if epoch % args.test_interval == 1 or args.test_interval == 1:
            evaluate_task_queue.put((deepcopy(model), device, epoch, 0, test_dataloader, args, loss_func))
            logger.info(f"len(evaluate_task_queue)={evaluate_task_queue.qsize()}")

            if epoch != 1:
                # result = evaluate(deepcopy(model), device, epoch, 0, test_dataloader, args, loss_func)
                result, eval_epoch = evaluate_result_queue.get()
                logger.info(f"[epoch: {eval_epoch}] acc_1: {result[0]} acc_5: {result[1]} loss: {result[2]}")
                summary_writer.add_scalar("test_acc_1", result[0], eval_epoch)
                summary_writer.add_scalar("test_acc_5", result[1], eval_epoch)
                summary_writer.add_scalar("test_loss", result[2], eval_epoch)
                summary_writer.add_scalar("training_time_test_acc_1", result[0], int(total_simulated_epoch_time))
                summary_writer.add_scalar("training_time_test_acc_5", result[1], int(total_simulated_epoch_time))
                summary_writer.add_scalar("training_time_test_loss", result[2], int(total_simulated_epoch_time))

        total_simulated_epoch_time += simulated_epoch_time

        logger.info(f"[{epoch}] distribution:{distribution_time} calculation:{calculation_time} aggregation:{aggregation_time}")        
        logger.info(f"[{epoch}] indeed time using: {indeed_epoch_time}")
        logger.info(f"[{epoch}] simulated time using: {simulated_epoch_time}")

        summary_writer.add_scalar("indeed_epoch_time", indeed_epoch_time, epoch)
        summary_writer.add_scalar("simulated_epoch_time", simulated_epoch_time, epoch)
        summary_writer.add_scalar("average local distribution", average_local_distribution_time, epoch)

        distribution_time_list.append(distribution_time)
        summary_writer.add_scalar("distribution", distribution_time, epoch)
        average_local_distribution_time_list.append(average_local_distribution_time)
        summary_writer.add_scalar("local distribution", average_local_distribution_time, epoch)
        logger.info(f"finish distribution: {distribution_time}, average local distribution time: {average_local_distribution_time}")

        calculation_time_list.append(calculation_time)
        summary_writer.add_scalar("calculation", calculation_time, epoch)
        averate_local_calculation_time_list.append(average_local_calculation_time)
        summary_writer.add_scalar("local calculation", average_local_calculation_time, epoch)
        logger.info(f"finish calculation: {calculation_time}, average local calculation time: {average_local_calculation_time}")

        aggregation_time_list.append(aggregation_time)
        summary_writer.add_scalar("aggregation", aggregation_time, epoch)
        average_local_aggregation_time_list.append(average_local_aggregation_time)
        summary_writer.add_scalar("local aggregation", average_local_aggregation_time, epoch)
        logger.info(f"finish aggregation: {aggregation_time}, average local aggregation time: {average_local_aggregation_time}")

        logger.info(f"average distribution time: {sum(distribution_time_list)/len(distribution_time_list)}")
        logger.info(f"average local distribution time: {sum(average_local_distribution_time_list)/len(average_local_distribution_time_list)}")

        logger.info(f"average calculation time: {sum(calculation_time_list)/len(calculation_time_list)}")
        logger.info(f"average local calculation time: {sum(averate_local_calculation_time_list)/len(averate_local_calculation_time_list)}")

        logger.info(f"average aggregation time: {sum(aggregation_time_list)/len(aggregation_time_list)}")
        logger.info(f"average local aggregation time: {sum(average_local_aggregation_time_list)/len(average_local_aggregation_time_list)}")

        logger.info(f"average indeed epoch time: {total_indeed_epoch_time/epoch}")
        logger.info(f"average simulated epoch time: {total_simulated_epoch_time/epoch}")


def get_required_memory_size(args):
    if args.model == "resnet18" and args.dataset == "femnist":
        # 1000MB
        return 1000


def select_device(args)->torch.device:
    required_memory_mb = get_required_memory_size(args)
    if torch.cuda.is_available():
        try:
            # 尝试分配所需显存大小的空间来测试是否有足够的显存
            dummy_tensor = torch.empty(required_memory_mb * (1024**2), dtype=torch.uint8, device='cuda:0')
            del dummy_tensor  # 释放测试用的显存
            return torch.device('cuda')
        except Exception as e:
            print(e)
    else:
        print("no cuda")
    return torch.device('cpu')


def train_handler(args, logger, state_dict, train_dataloader):
    """
        return model, debug_param, total_loss, sample_number, calculation_start_time, calculation_stop_time, calculation_time
    """
    # move to gpu
    # the GPU memory may not be sufficient when many of the selected clients use the GPU
    # we try to allocate 1G memory in GPU. If this fails, we use GPU to train.
    # This process happens in each epoch.
    if args.use_gpu:
        device = d2l.try_gpu()
    else:
        device = torch.device("cpu")
        
    # init model
    # model = models.get_model(args.model, args.dataset, args.num_classes)
    model = fedscale_dataset.init_model()
    model.load_state_dict(state_dict)
    model = model.to(device)
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
    sample_number = 0
    # create an empty tensor to store all samples' loss
    total_loss = torch.Tensor().to(device)
    model.train()
    calculation_start_time = time.time()
    
    for pwd_iter, (x, y) in enumerate(train_dataloader):
        # Here, clients are computing.
        pwd_iter += 1
        sample_number += len(y)
        optimizer.zero_grad()
        x, y = x.to(device), y.to(device)
        y_hat = model(x)
        losses = loss_func(y_hat, y)
        losses.mean().backward()
        for param in model.parameters():
            if param.requires_grad:
                logger.debug(f"param.grad[0][0]: {param.grad[0][0]}")
                break
        optimizer.step()
        total_loss = torch.cat((total_loss, losses.detach().clone()))
        logger.info(f"{pwd_iter}")

    calculation_stop_time = time.time()
    calculation_time = calculation_stop_time - calculation_start_time
    logger.info(f"actual calculation time: {calculation_time}")

    return model, debug_param, total_loss, sample_number, calculation_start_time, calculation_stop_time, calculation_time


def train(args, logger, state_dict, train_dataloader):
    # If we use GPU and the GPU memory is insufficient, wait until it's sufficient.
    while True:
        try:
            return train_handler(args, logger, state_dict, train_dataloader)
        except Exception as e:
            logger.warning(e)
            time.sleep(10)


def ping(logger):
    StackName = os.getenv("StackName")
    hostname = os.getenv("Hostname")
    computaion_node_domain_name = f"{StackName}_computation_node_{hostname}"

    while True:
        try:
            response = requests.get(f"http://{computaion_node_domain_name}:27500/ping", timeout=10)
        except Exception as e:
            logger.warning(e)
        else:
            logger.info(f"ping ok {response}")
            break

# def request_computation_node(args, input_state_dict_path, pwd_rank, logger):
def request_computation_node(args, input_state_dict, pwd_rank, logger):
    # {StackName}_computation_node_{id} id from 0 to the number of physical machines
    StackName = os.getenv("StackName")
    hostname = os.getenv("Hostname")
    # PhysicalClientMachineNumber = int(os.getenv("PhysicalClientMachineNumber"))
    # id = random.choice(list(range(PhysicalClientMachineNumber)))
    # {StackName}_computation_node_{id} id from 0 to the number of physical machines
    computaion_node_domain_name = f"{StackName}_computation_node_{hostname}"
    # params = {
    #     "args": vars(args),
    #     "input_state_dict_path": input_state_dict_path,
    #     "pwd_rank": pwd_rank,
    # }

    params = {
        "args": vars(args),
        "input_state_dict": utils.state_dict_base64_encode(input_state_dict),
        "pwd_rank": pwd_rank,
    }
    ping(logger)
    while True:
        try:
            response = requests.post(f"http://{computaion_node_domain_name}:27500/compute", json=params, timeout=600)
            result = response.json()
        except (RemoteDisconnected, ProtocolError, ConnectionError, ReadTimeout, JSONDecodeError) as e:
            logger.warning(e)
            time.sleep(10)
        except Exception as e:
            logger.warning(e)
            exit(0)
        else:
            break
    return result

async def run_client(args, logger):
    client = grpc_client.GrpcClient(args, logger)
    await client.communicate()

def client_func(args, logger:logging.Logger):
    RANK, WORLD_SIZE = dist.get_rank(), dist.get_world_size()

    config = utils.update_config_file(RANK, args.master_addr, args.master_port, args.model, args.dataset, logger)
    model_root_dir = config.model.ModelPath
    os.makedirs(model_root_dir, exist_ok=True)

    log_path =  os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    log_path = os.path.join(log_path, "logs", os.getenv("DateTime"))

    if args.use_gpu:
        # preferred device. May not actually run on this device for lack of GPU memory.
        device = d2l.try_gpu()
    else:
        device = torch.device("cpu")

    if args.dataset == "femnist":
        participants_number = 2800
    elif args.dataset == "google_speech_commands":
        participants_number = 2167
    elif args.dataset == "openimage":
        participants_number = 11325
    else:
        raise ValueError
    logger.info(f"participants_number {participants_number}")

    if args.running_mode == utils.SIMULATION_MODE:
        client_profiles = [None for _ in range(1+participants_number)]
        dist.broadcast_object_list(client_profiles, 0)
        logger.info(f"client_profiles {client_profiles}")

    # dataset
    # training_sets = fedscale_dataset.get_train_datasets(logger)
    # train_dataset, train_dataloader, test_dataset, test_dataloader = datasets.load_dataset(
    #     args.dataset, args.data_dir, args.batch_size, args
    # )
    # if args.dataset != "mnist":
    #     sampler = DistributedSampler(train_dataset, args.world_size-1, RANK-1)
    #     train_dataloader = DataLoader(train_dataset, args.batch_size, shuffle=False, sampler=sampler)

    # dataset_index = scatter_python_objects()
    # logger.info(f"train data length: {len(dataset_index)}")
    # 加载训练数据集
    # train_dataloader = fedscale_dataset.get_client_train_dataloader(dataset_index)
    # for x,y in train_dataloader:
    #     logger.info(f"batch shape: {x.shape}, {y.shape}")
    #     break
    # logger.info(next(iter(train_dataloader))[0].shape)

    if args.transfer_mode == utils.BTPS_TRANSFER_MODE:
        torrent = None
        path = None
        tc = TorrentCommunicationPyTorch(logger)

    for epoch in range(1, args.num_epochs+1):
        dist.barrier()
        # 传输该docker需要模拟的client index/rank
        pwd_rank = transfer_client_selection_result()
        logger.info(f"[{epoch}] {RANK} simulate {pwd_rank}")

        if args.running_mode == utils.SIMULATION_MODE:
            if args.rate_limit:
                utils.rate_limit(RANK, client_profiles[pwd_rank], logger)

        # distribution
        # recv the model to CPU
        if args.transfer_mode == utils.BTPS_TRANSFER_MODE:
            path = os.path.join(model_root_dir,  f"{epoch}.pth")
            logger.debug(path)
            
            dist.barrier()
            distribution_start_time = time.time()
            torrent = tc.bt_broadcast(None)
            downloading_output = tc.bt_recv(torrent)
            logger.info(f"downloading_output {downloading_output}")
            distribution_stop_time = time.time()
            distribution_time = distribution_stop_time - distribution_start_time

            # load            
            # state_dict = torch.load(path, map_location=device)
            state_dict = torch.load(path)
            
        elif args.transfer_mode == utils.PS_TRANSFER_MODE:
            # # params broadcast
            # # for params in model.parameters():
            # state_dict = dict()
            # for name, params in model.state_dict().items():
            #     tmp = torch.zeros_like(params, device=torch.device("cpu"))
            #     start = time.time()
            #     dist.broadcast(tmp, 0, group=sub_group)
            #     distribution_time += (time.time()-start)
            #     state_dict[name] = tmp.to(device)

            # # state_dict broadcast
            # state_dict_tensor = utils.broadcast_unfixed_length_tensor(None, 0, sub_group)
            # state_dict = utils.python_object_to_tensor(state_dict_tensor, reverse=True)
            # for name, params in state_dict.items():
            #     state_dict[name] = params.to(device)

            # state_dict send/recv
            # size
            dist.barrier()
            distribution_start_time = time.time()
            size = torch.empty(1, dtype=torch.int64)
            dist.broadcast(size, 0)
            # data
            # recv synchronously to measure the distribution time
            state_dict_tensor = torch.empty(size[0], dtype=torch.uint8)
            dist.recv(state_dict_tensor, 0)
            distribution_stop_time = time.time()
            distribution_time = distribution_stop_time - distribution_start_time

            state_dict = utils.python_object_to_tensor(state_dict_tensor, reverse=True)
            for name, params in state_dict.items():
                # state_dict[name] = params.to(device)
                state_dict[name] = params

        logger.info(f"finish distribution: {distribution_time}")

        # model, debug_param, total_loss, sample_number, calculation_start_time, calculation_stop_time, calculation_time = train(args, logger, state_dict, train_dataloader)
        # input_state_dict_path = os.path.join(log_path, f"{pwd_rank}_input.pth")
        # torch.save(state_dict, input_state_dict_path)
        # ret = request_computation_node(args, input_state_dict_path, pwd_rank, logger)
        # os.remove(input_state_dict_path)
        ret = request_computation_node(args, state_dict, pwd_rank, logger)
        logger.info("finish indeed calculation")
        statistical_utility, sample_number, calculation_start_time, calculation_stop_time, calculation_time = ret['statistical_utility'], ret['sample_number'], ret['calculation_start_time'], ret['calculation_stop_time'], ret['calculation_time']
        # while True:
        #     try:
        #         ret = request_computation_node(args, input_state_dict_path, dataset_index, logger)
        #     except Exception as e:
        #         logger.warning(e)
        #         time.sleep(10)
        #     else:
        #         statistical_utility, sample_number, calculation_start_time, calculation_stop_time, calculation_time = ret['statistical_utility'],  ret['sample_number'], ret['calculation_start_time'], ret['calculation_stop_time'], ret['calculation_time']
        #         break
        # statistical_utility = calculate_statistical_utility(total_loss, sample_number)

        # # init model
        # # model = models.get_model(args.model, args.dataset, args.num_classes)
        # model = fedscale_dataset.init_model()
        # model = model.to(device)
        # model.load_state_dict(state_dict)
        # del state_dict
        # # set reduction to none to keep each sample's loss
        # loss_func = nn.CrossEntropyLoss(reduction='none')
        # # loss_func = nn.CrossEntropyLoss()
        # # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
        # # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        # # optimizer = torch.optim.Adam(model.parameters())
        # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

        # logger.info(f"finish distribution: {distribution_time}")

        # # # evaluation
        # # result = evaluate(deepcopy(model), device, epoch, 0, test_dataloader, nn.CrossEntropyLoss())
        # # logger.info(f"[epoch: {epoch}] acc: {result[0]} loss: {result[1]}")

        # # calculation
        # debug_param = next(model.parameters())
        # logger.debug(f"before train, param:\n{debug_param[0][0]}")
        # model.train()
        # sample_number = 0
        # # create an empty tensor to store all samples' loss
        # total_loss = torch.Tensor().to(device)
        # model.train()
        # calculation_start_time = time.time()
        
        # for pwd_iter, (x, y) in enumerate(train_dataloader):
        #     # Here, clients are computing.
        #     pwd_iter += 1
        #     sample_number += len(y)
        #     optimizer.zero_grad()
        #     x, y = x.to(device), y.to(device)
        #     y_hat = model(x)
        #     losses = loss_func(y_hat, y)
        #     losses.mean().backward()
        #     for param in model.parameters():
        #         if param.requires_grad:
        #             logger.debug(f"param.grad[0][0]: {param.grad[0][0]}")
        #             break
        #     optimizer.step()
        #     total_loss = torch.cat((total_loss, losses.detach().clone()))
        #     logger.info(f"{pwd_iter}")

        # calculation_stop_time = time.time()
        # calculation_time = calculation_stop_time - calculation_start_time
        # logger.info(f"actual calculation time: {calculation_time}")
        # In simulation mode, the actually-measured calculation time is not accurate.
        # It should be calculated from client profile and then sleep for a while.
        if args.running_mode == utils.SIMULATION_MODE:
            # 这个barrier是让实际计算时间的差异被抵消
            dist.barrier()
            # simulated distribution time
            if not args.quick_simulate:
                time.sleep(distribution_time)
            calculation_start_time = time.time()
            calculation_time = 3 * args.local_epoch * sample_number * client_profiles[pwd_rank]['computation'] / 1000.0 / 10
            calculation_stop_time = calculation_start_time + calculation_time
            logger.info(f"simulated calculation_time: {calculation_time}")
            if not args.quick_simulate:
                time.sleep(calculation_time if calculation_time < 1000 else 1000)
                # time.sleep(calculation_time)
        else:
            logger.info(f"finish calculation: {calculation_time}")

        # aggregation
        # # params broadcast
        # # for params in model.parameters():
        # for name, params in model.state_dict().items():
        #     # tmp = params.data.cpu()
        #     tmp = params.cpu()
        #     start = time.time()
        #     dist.reduce(tmp, 0, op=dist.ReduceOp.SUM, group=sub_group)
        #     aggregation_time += (time.time()-start)

        # # # state_dict broadcast
        # state_dict = dict()
        # for name, params in model.state_dict().items():
        #     state_dict[name] = params.cpu()
        # state_dict_tensor = utils.python_object_to_tensor(state_dict)
        # dist.gather(state_dict_tensor, None, 0, group=sub_group)

        # state_dict send/recv
        # state_dict = dict()
        # for name, params in model.state_dict().items():
        #     state_dict[name] = params.cpu()
        # state_dict = torch.load(ret['output_state_dict_path'])
        # os.remove(ret['output_state_dict_path'])
        state_dict = utils.state_dict_base64_decode(ret['output_state_dict'])
        dataset_length_tensor = utils.python_object_to_tensor(f"{sample_number:0{6}d}")
        state_dict_tensor = utils.python_object_to_tensor(state_dict)
        data_tensor = torch.cat([dataset_length_tensor, state_dict_tensor])
        logger.info(f"{state_dict_tensor[:10]} {state_dict_tensor[-10:]} {dataset_length_tensor} {data_tensor[:20]}")
        # send synchronously
        logger.info(f"data_tensor.size() {data_tensor.size()}")

        aggregation_start_time = time.time()
        dist.send(data_tensor, 0)
        aggregation_stop_time = time.time()
        aggregation_time = aggregation_stop_time - aggregation_start_time
        logger.info(f"finish aggregation: {aggregation_time}")

        # TODO: transfer messages for selection
        # DONE
        # 1) In all situations, use each client's selection_messages to get global calculation_time and aggregation_time.
        # 2) in oort, selection_messages are required to get the statistical utility.
        # RANK's range varies from 0 to WORLD_SIZE-1. Its byte length is not fixed. Convert it to string.

        if args.transfer_mode == utils.BTPS_TRANSFER_MODE:
            stop_seeding_output = tc.stop_seeding(torrent)
            logger.info(f"contribution: {stop_seeding_output['byteswrittendata']}")
            contribution = stop_seeding_output['byteswrittendata'] 
            os.remove(path)
        else:
            contribution = 0

        selection_message = {
            "rank": f"{pwd_rank:0{len(str(participants_number))}d}",
            # As the shape of loss tensor is not fixed on different clients, we just send the statistical utility calculated by each client.
            "statistical_utility": statistical_utility,
            "rate": client_profiles[pwd_rank]['communication'],
            "distribution_time": distribution_time,
            "distribution_start_time": distribution_start_time,
            "distribution_stop_time": distribution_stop_time,
            "calculation_time": calculation_time,
            "calculation_start_time": calculation_start_time,
            "calculation_stop_time": calculation_stop_time,
            "aggregation_time": aggregation_time,
            "aggregation_start_time": aggregation_start_time,
            "aggregation_stop_time": aggregation_stop_time,
            "contribution": f"{contribution:0{10}d}",
        }
        selection_message_tensor = utils.python_object_to_tensor(selection_message)
        # size
        size = torch.tensor([selection_message_tensor.shape[0]], dtype=torch.int64)
        logger.debug(f"selection_message_tensor.size: {size}")
        dist.gather(size, None, 0)
        logger.info(f"selection_message: {selection_message}, shape: {selection_message_tensor.shape}")
        dist.gather(selection_message_tensor, None, 0)

        # for params in model.parameters():
        #     logger.debug(f"params.sum(): {params.sum()}")
        #     break

        # logger.debug(f"after aggregation, param:\n{debug_param[0][0]}")
        logger.info(f"[{epoch}] aggregation:{aggregation_time} distribution:{distribution_time}")

        # 节约显存或内存，清除模型
        # del optimizer
        # del model
        del state_dict
        del state_dict_tensor
        # del total_loss
        torch.cuda.empty_cache()


def main(args):
    rank = args.rank
    logger = utils.get_logger(args, f"[{rank}]")
    logger.propagate = False

    if args.running_mode == utils.SIMULATION_MODE:
        logger.info("Run in Simulation Mode.")

    logger.info(f"command line arguments: {args}")
    
    if rank == 0:
        asyncio.run(run_server(args, logger))
    else:
        asyncio.run(run_client(args, logger))


if __name__ == "__main__":
    args = utils.get_args()
    print("args", args)
    print("model, dataset", args.model, args.dataset)
    fedscale_dataset.set_yaml_conf_to_parser(args.model, args.dataset)

    if args.rank == 0:
        # model_name, dataset_name = "resnet18", "femnist"
        # model_name, dataset_name = "shufflenet_v2_x2_0", "amazon"
        # model_name, dataset_name = "resnet34", "google_speech_commands"
        # model_name, dataset_name = 'shufflenet_v2_x2_0', 'openimage'

        model = fedscale_dataset.load_model()
        print(model)
        tmp_path = "test.pth"
        torch.save(model.state_dict(), tmp_path)
        print("model size", os.stat("test.pth").st_size/1024.0/1024.0)
        os.remove(tmp_path)
        
        clients_data, test_dataloader = fedscale_dataset.load_dataset(client_num=args.world_size - 1)

        device = torch.device("cuda:0")
        model = model.cuda()
        for x, y in test_dataloader:
            if args.dataset == "google_speech_commands":
                x, y = torch.unsqueeze(x, 1).to(device), y.to(device)
            else:
                x, y = x.to(device), y.to(device)
                
            print(x.shape)
            y_hat = model(x)
            print(y_hat,y)
            break

    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = args.master_port
    
    # while True:
    #     try:
    #         dist.init_process_group("gloo", rank=args.rank, world_size=args.world_size)
    #         break
    #     except Exception as e:
    #         print(e)
    #         time.sleep(1)
    
    utils.set_seed(args.seed)
    main(args)
