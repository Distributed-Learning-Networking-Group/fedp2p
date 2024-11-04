import sys
sys.path.append("/app/bt_ps")
sys.path.append("/app/bt_ps/thirdparty/FedScale")
print(sys.path)
import threading
import torch
import time
import torch.nn as nn
import os
import pytz

from d2l import torch as d2l
from flask import Flask, request, make_response, jsonify
from copy import deepcopy
from datetime import datetime
from gunicorn.app.base import BaseApplication

from fedscale.utils import dataset as fedscale_dataset
from fedscale.cloud.fllibs import *
from p2p_server.utils import utils
from p2p_server.models import models

app = Flask(__name__)
training_sets = None
# 并发控制
concurrent = 1
semaphore=threading.Semaphore(concurrent)
log_path =  os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
log_path = os.path.join(log_path, "logs", os.getenv("DateTime"))


@app.route('/ping', methods=['GET'])
def ping():
    return "ok"

beijing_tz = pytz.timezone('Asia/Shanghai')
def timestamp_to_datetime(t:float):
    return datetime.fromtimestamp(t, beijing_tz)

def train_handler(args, logger, state_dict, train_dataloader):
    """
        return model, debug_param, total_loss, sample_number, calculation_start_time, calculation_stop_time, calculation_time
    """
    # move to gpu
    # the GPU memory may not be sufficient when many of the selected clients use the GPU
    # we try to allocate 1G memory in GPU. If this fails, we use GPU to train.
    # This process happens in each epoch.
    if args.use_gpu and torch.cuda.is_available():
        device = os.getenv("device")
        if device is not None: # 没有指定的设备
            device = torch.device(device)
        else: # 从可选设备中进行尝试
            device = d2l.try_gpu()
    else: # 不使用GPU或GPU不可用
        device = torch.device("cpu")
        
    # init model
    # model = models.get_model(args.model, args.dataset, args.num_classes)
    model = fedscale_dataset.init_model()
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.train()

    if args.gradient_policy == utils.AggregateType.FEDPROX_STRATEGY.value:
        global_weights = deepcopy(model.state_dict())

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
    
    # create an empty tensor to store all samples' loss
    total_loss = torch.Tensor().to(device)
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

            losses = loss_func(y_hat, y)
            if args.gradient_policy == utils.AggregateType.FEDPROX_STRATEGY.value:
                proximal_term = 0.0
                for w, w_t in zip(model.parameters(), global_weights.values()):
                    proximal_term += (w - w_t).norm(2)
            elif args.gradient_policy == utils.AggregateType.FEDAVG_STRATEGY.value:
                proximal_term = 0
            loss = losses.mean() + (args.proxy_mu / 2) * proximal_term
            loss.backward()
            
            for param in model.parameters():
                if param.requires_grad:
                    # logger.debug(f"param.grad[0][0]: {param.grad[0][0]}")
                    break
            optimizer.step()
            total_loss = torch.cat((total_loss, losses.detach().clone()))

    calculation_stop_time = time.time()
    calculation_time = calculation_stop_time - calculation_start_time
    # logger.info(f"actual calculation time: {timestamp_to_datetime(calculation_stop_time)} - {timestamp_to_datetime(calculation_start_time)} = {calculation_time}")
    statistical_utility = utils.calculate_statistical_utility(total_loss, sample_number)

    return model, debug_param, statistical_utility, sample_number, calculation_start_time, calculation_stop_time, calculation_time


def train(args, logger, state_dict, train_dataloader):
    # If we use GPU and the GPU memory is insufficient, or other exceptions are raised, 
    # just return the exception message and let the client retry.
    try:
        semaphore.acquire()
        res = train_handler(args, logger, state_dict, train_dataloader)
    except Exception as e:
        logger.warning(e)
        res = None
    finally:
        semaphore.release()
        return res

@app.route('/compute', methods=['POST'])
def compute():
    """
        {
            "args": args,
            "state_dict": str (path to the state_dict), {rank}_input.pth
            "data_index": list,
        }

        {
            "state_dict": str (path to the state_dict), {rank}_output.pth
        }
    """
    # 获取请求中的数据
    global training_sets
    
    config = request.json
    args = utils.to_namespace(config['args'])
    pwd_rank = config['pwd_rank']
    logger = utils.get_logger(args, f"[{pwd_rank}]")
    logger.propagate = False
    # logger.info("in train")
    
    if training_sets is None:
        training_sets = fedscale_dataset.get_train_datasets(logger)

    train_dataloader = fedscale_dataset.select_dataset(
        pwd_rank, training_sets,
        batch_size=parser.args.batch_size, args=parser.args,
        collate_fn=fedscale_dataset.get_collate_fn()
    )
    # logger.info("load train dataloader")

    # input_state_dict = torch.load(config['input_state_dict_path'])
    input_state_dict = utils.state_dict_base64_decode(config['input_state_dict'])
    ret = train(args, logger, input_state_dict, train_dataloader)
    if ret is None:
        response = make_response(jsonify({"error": "Internal Server Error"}), 500)
    else:
        model, debug_param, statistical_utility, sample_number, calculation_start_time, calculation_stop_time, calculation_time = ret
        model = model.to("cpu")
        ret = {
            # "output_state_dict_path": output_state_dict_path,
            "output_state_dict": utils.state_dict_base64_encode(model.state_dict()),
            "statistical_utility":statistical_utility,
            "sample_number":sample_number,
            "calculation_start_time":calculation_start_time,
            "calculation_stop_time":calculation_stop_time,
            "calculation_time":calculation_time,
        }
        response = make_response(jsonify(ret), 200)
        logger.info(f"client port: {request.remote_addr}, statistical_utility: {ret['statistical_utility']}, sample_number: {ret['sample_number']}, calculation_start_time: {ret['calculation_start_time']}, calculation_stop_time: {ret['calculation_stop_time']}, calculation_time: {ret['calculation_time']}")

        return response
    # if data is None:
        # return jsonify({"error": "No data provided"}), 400
    
    # try:
    #     # 转换数据为整数
    #     data = int(data)
    #     # 执行计算
    #     result_path = perform_computation(data)
    #     # 返回计算结果文件的路径
    #     return jsonify({"result_path": result_path})
    # except ValueError:
    #     return jsonify({"error": "Invalid data. Please provide an integer."}), 400

def run_uvicorn():
    import subprocess
    cmd = [
        "uvicorn",
        "ps_param_param_epoch_cs_computation:app",
        "--host", "0.0.0.0",
        "--port", "27500",
        "--workers", "4",
    ]
    subprocess.run(cmd)


class GunicornApp(BaseApplication):
    def __init__(self, app, options=None):
        self.app = app
        self.options = options or {}
        super().__init__()

    def load_config(self):
        config = {
            key: value for key, value in self.options.items()
            if key in self.cfg.settings and value is not None
        }
        for key, value in config.items():
            self.cfg.set(key.lower(), value)

    def load(self):
        return self.app
if __name__ == '__main__':
    args = utils.get_args()
    fedscale_dataset.set_yaml_conf_to_parser(args.model, args.dataset)
    
    # options = {
    #     'bind': '0.0.0.0:27500',
    #     'workers': 5,
    #     'threads': 2
    # }
    # GunicornApp(app, options).run()
    # app.run(debug=False, host='0.0.0.0', port=27500, threaded=True)
    app.run(debug=False, host='0.0.0.0', port=27500)
    