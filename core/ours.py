import json
import os
import copy
import time
import math
import torch
import random
import argparse
from models.CLIP import *
from utils.get_data import domainnet
from utils.get_data import get_data
from utils.data_utils import build_subset, split_train_and_val
from utils.server import Server
from utils.clientours import Client
from utils.json_utils import generate_json_config
import warnings
import numpy as np
from collections import defaultdict
warnings.simplefilter("ignore")

torch.manual_seed(1)
torch.cuda.manual_seed(1) if torch.cuda.is_available() else None


def generate_protos_training_data(uploaded_protos, batchsize=10):
    classes = uploaded_protos[0].keys()
    protos = []
    labels = []

    for proto in uploaded_protos:
        for c in classes:
            protos_class_c = proto[c]
            protos.append(protos_class_c)
            labels.extend([c] * protos_class_c.shape[0])

    protos = torch.vstack(protos)
    labels = torch.tensor(labels, dtype=torch.long)

    total_protos = protos.shape[0]
    perm = torch.randperm(total_protos)
    protos = protos[perm]
    labels = labels[perm]

    max_full_batches = total_protos // batchsize
    new_total = max_full_batches * batchsize

    protos = protos[:new_total]
    labels = labels[:new_total]

    protos = protos.view(-1, batchsize, protos.shape[-1])
    labels = labels.view(-1, batchsize)

    training_data = []
    for i in range(protos.shape[0]):
        training_data.append((protos[i], labels[i]))

    return training_data

def send_adaptive_global_adapter(global_adapter, clientObjs):
    for client in clientObjs:
        client.model.base.global_adapter.load_state_dict(global_adapter.state_dict())
        client.model.base.adapter.load_state_dict(global_adapter.state_dict())
    return clientObjs

def send_global_head(global_cls_head, clientObjs):
    for client in clientObjs:
        client.model.head.load_state_dict(global_cls_head.state_dict())
    return clientObjs

def server_adative_training(training_data, server, threshold=0.001, num_losses=20):
    losses = []
    server.image_encoder.train()
    server.global_cls_head.train()
    server.freeze_except_global_adapter()
    optimizer = torch.optim.AdamW(server.image_encoder.global_adapter.parameters(), lr=server.learning_rate)
    def lr_lambda(current_epoch):
        if current_epoch < server.warm_up:
            return (float(current_epoch) + 1) / float(max(1, server.warm_up))
        else:
            # Cosine annealing
            return 0.5 * (1 + math.cos(math.pi * (current_epoch - server.warm_up) / (server.max_epochs - server.warm_up)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    convergence_epochs = 0
    while True:
        for i, (proto, label) in enumerate(training_data):
            # print('proto:', proto.shape)
            # print('label:', label.shape)
            optimizer.zero_grad()
            proto = proto.to(server.device)
            label = label.to(server.device)
            rep = server.image_encoder.global_adapter(proto)
            rep = rep + proto
            output = server.global_cls_head(rep)
            loss = server.criterion(output, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(server.image_encoder.global_adapter.parameters(), 1)
            optimizer.step()

            losses.append(loss.item())
        convergence_epochs += 1
        scheduler.step()
        print(f'server epoch {convergence_epochs} loss std: {np.std(losses[-num_losses:])}')
        if np.std(losses[-num_losses:]) < threshold and len(losses) > num_losses:
            print(f'convergence at epoch {convergence_epochs}')
            break

        if convergence_epochs >= server.max_epochs:
            print(f'exceed max epochs {server.max_epochs}')
            break

    train_metrics = {
        "server_epochs": convergence_epochs,
        "loss_last": float(losses[-1]) if losses else None,
        "loss_std_last": float(np.std(losses[-num_losses:])) if losses else None,
        "num_batches": len(training_data),
    }

    return server.image_encoder.global_adapter, train_metrics

def receive_protos(clients):
    uploaded_ids = []
    uploaded_protos = []
    for client in clients:
        uploaded_ids.append(client.id)
        uploaded_protos.append(client.protos)
    return uploaded_protos

def save_jsonl(path, record):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a+", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False)
        f.write("\n")

def log_uploaded_proto_stats(uploaded_protos, save_path=None):
    """
    uploaded_protos:
        list[dict[label -> Tensor[num_proto, dim]]]
    """
    stats = {
        "num_clients": len(uploaded_protos),
        "clients": [],
        "total_num_protos": 0,
    }

    for client_id, proto_dict in enumerate(uploaded_protos):
        client_stat = {
            "client_id": client_id,
            "num_classes": len(proto_dict),
            "class_stats": {},
            "num_protos": 0,
        }

        for label, protos in proto_dict.items():
            if protos.dim() == 1:
                protos = protos.unsqueeze(0)
            protos_cpu = protos.detach().float().cpu()
            num_proto = int(protos_cpu.shape[0])
            dim = int(protos_cpu.shape[1])

            norms = protos_cpu.norm(dim=1)
            class_stat = {
                "num_proto": num_proto,
                "dim": dim,
                "mean_norm": float(norms.mean().item()),
                "std_norm": float(norms.std().item()) if num_proto > 1 else 0.0,
            }

            client_stat["class_stats"][int(label)] = class_stat
            client_stat["num_protos"] += num_proto

        stats["total_num_protos"] += client_stat["num_protos"]
        stats["clients"].append(client_stat)

    print(
        f"[ProtoStats] clients={stats['num_clients']}, "
        f"total_protos={stats['total_num_protos']}"
    )

    if save_path is not None:
        save_jsonl(save_path, {"type": "uploaded_proto_stats", "stats": stats})

    return stats

def get_text_anchors_from_global_head(server):
    """
    使用 global classification head 的权重作为 text anchors。
    注意：global_cls_head.weight 可能带 logit_scale，因此这里做 normalize。
    """
    text_anchors = server.global_cls_head.weight.detach().clone()
    text_anchors = torch.nn.functional.normalize(text_anchors.float(), dim=-1)
    return text_anchors

def compute_proto_text_drift(uploaded_protos, text_anchors):
    """
    计算每个 client / class 的 prototype-text drift.

    return:
        drift_stats: dict
    """
    text_anchors = text_anchors.detach().float().cpu()
    text_anchors = torch.nn.functional.normalize(text_anchors, dim=-1)

    all_drifts = []
    client_drifts = {}

    for client_id, proto_dict in enumerate(uploaded_protos):
        per_client_drifts = []
        per_class = {}

        for label, protos in proto_dict.items():
            label_int = int(label)

            if protos.dim() == 1:
                protos = protos.unsqueeze(0)
            p = protos.detach().float().cpu()
            p = torch.nn.functional.normalize(p, dim=-1)

            if label_int >= text_anchors.shape[0]:
                print(f"[Warning] label {label_int} out of text_anchors range")
                continue

            t = text_anchors[label_int].view(1, -1)

            cosine = (p * t).sum(dim=-1)
            drift = 1.0 - cosine

            mean_drift = float(drift.mean().item())
            std_drift = float(drift.std().item()) if drift.numel() > 1 else 0.0

            per_class[label_int] = {
                "mean": mean_drift,
                "std": std_drift,
                "num_proto": int(p.shape[0]),
            }

            per_client_drifts.extend(drift.tolist())
            all_drifts.extend(drift.tolist())

        client_drifts[client_id] = {
            "mean": float(np.mean(per_client_drifts)) if per_client_drifts else 0.0,
            "std": float(np.std(per_client_drifts)) if per_client_drifts else 0.0,
            "per_class": per_class,
        }

    drift_stats = {
        "proto_text_drift_mean": float(np.mean(all_drifts)) if all_drifts else 0.0,
        "proto_text_drift_std": float(np.std(all_drifts)) if all_drifts else 0.0,
        "proto_text_drift_max": float(np.max(all_drifts)) if all_drifts else 0.0,
        "proto_text_drift_min": float(np.min(all_drifts)) if all_drifts else 0.0,
        "client_drifts": client_drifts,
    }

    return drift_stats

def log_drift_stats(drift_stats, save_path=None):
    print(
        "[DriftStats] "
        f"mean={drift_stats['proto_text_drift_mean']:.6f}, "
        f"std={drift_stats['proto_text_drift_std']:.6f}, "
        f"min={drift_stats['proto_text_drift_min']:.6f}, "
        f"max={drift_stats['proto_text_drift_max']:.6f}"
    )

    if save_path is not None:
        save_jsonl(save_path, {"type": "drift_stats", "stats": drift_stats})

def refine_uploaded_protos(uploaded_protos, text_anchors=None, method="none", **kwargs):
    """
    当前是占位函数。
    后续可以在这里加入：
    1. neighbor refinement
    2. residual GCN
    3. text-anchor graph refinement

    现在 method='none' 时，不修改 uploaded_protos。
    """
    graph_metrics = {
        "graph_refine_method": method,
        "enabled": False,
        "num_nodes": 0,
        "num_edges": 0,
        "message": "No graph refinement is applied.",
    }

    if method is None or method == "none":
        return uploaded_protos, graph_metrics

    raise NotImplementedError(
        f"Graph refinement method '{method}' is not implemented yet."
    )

def log_graph_metrics(graph_metrics, save_path=None):
    print(
        "[GraphMetrics] "
        f"method={graph_metrics.get('graph_refine_method')}, "
        f"enabled={graph_metrics.get('enabled')}, "
        f"nodes={graph_metrics.get('num_nodes')}, "
        f"edges={graph_metrics.get('num_edges')}"
    )

    if save_path is not None:
        save_jsonl(save_path, {"type": "graph_metrics", "stats": graph_metrics})


def proto_aggregation(local_protos_list):
    agg_protos_label = defaultdict(list)
    for local_protos in local_protos_list:
        for label in local_protos.keys():
            agg_protos_label[label].append(local_protos[label])

    for [label, proto_list] in agg_protos_label.items():
        if len(proto_list) > 1:
            proto = 0 * proto_list[0].data
            for i in proto_list:
                proto += i.data
            agg_protos_label[label] = proto / len(proto_list)
        else:
            agg_protos_label[label] = proto_list[0].data

    return agg_protos_label

def calculate_fedts_weights(clients):
    # every client use the same weight
    weights = [1/len(clients) for c in clients]
    return weights

def proto_initialization(clientObjs, server, args=None):
    uploaded_protos = receive_protos(clientObjs)

    log_path = None
    if args is not None:
        log_path = (
            f"./results/debug/"
            f"{args.image_encoder_name}_{args.dataset}_sub{args.subset_size}_"
            f"sra{args.sample_ratio}_sram{args.sample_ratio_method}_debug.jsonl"
        )

    # 1. 记录上传 prototype 的统计信息
    log_uploaded_proto_stats(uploaded_protos, save_path=log_path)

    # 2. 从 global head 中取 text anchors
    text_anchors = get_text_anchors_from_global_head(server)

    # 3. 计算 prototype-text drift
    drift_stats = compute_proto_text_drift(uploaded_protos, text_anchors)
    log_drift_stats(drift_stats, save_path=log_path)

    # 4. 预留图校准入口。当前 method='none'，不做真正修改
    uploaded_protos, graph_metrics = refine_uploaded_protos(
        uploaded_protos,
        text_anchors=text_anchors,
        method="none",
    )
    log_graph_metrics(graph_metrics, save_path=log_path)

    # 5. 构造 prototype training data
    # global_protos = proto_aggregation(uploaded_protos) # do not aggregate the protos !!!
    training_data = generate_protos_training_data(uploaded_protos)

    # 6. 训练 global adapter
    global_adapter, server_metrics = server_adative_training(training_data, server)

    # 7. 下发 global adapter 和 global head
    clientObjs = send_adaptive_global_adapter(global_adapter, clientObjs)
    clientObjs = send_global_head(server.global_cls_head, clientObjs)
    server.image_encoder.global_adapter.load_state_dict(global_adapter.state_dict())

    return clientObjs, server

def calculate_fedavg_weights(clients):
    total_train_num = 0
    num_list = []
    for c in clients:
        train_num = len(c.train_dataloader) * c.batch_size
        total_train_num += train_num
        num_list.append(train_num)
    weights = [num/total_train_num for num in num_list]
    return weights

def fedavg(weights, clientObjs, server):
    print("FedAvg... with weights: ", weights)
    # server receive the adapters from clients
    adapters = [c.model.base.adapter for c in clientObjs]

    # fedavg aggregation
    server_global_adapter = copy.deepcopy(server.image_encoder.global_adapter)
    for param in server_global_adapter.parameters():
        param.data.zero_()

    for adapter in adapters:
        for w, global_param, param in zip(weights, server_global_adapter.parameters(), adapter.parameters()):
            global_param.data += w * param.data.clone()
    # set the global adapter to the server
    server.image_encoder.global_adapter.load_state_dict(server_global_adapter.state_dict())

    # send the global adapter back to the clients
    # param will be covered as global param
    for id in range(len(clientObjs)):
        for param, global_param in zip(clientObjs[id].model.base.adapter.parameters(), server_global_adapter.parameters()):
            param.data = global_param.data.clone()

    return clientObjs, server

def calculate_summary_acc(client_acc, clients):
    acc_matrix = np.array(client_acc, dtype=float)
    test_nums = np.array([len(client.test_dataset) for client in clients], dtype=float)
    ind_acc = np.sum(np.diag(acc_matrix) * test_nums) / np.sum(test_nums)
    client_avg_accs = np.sum(acc_matrix * test_nums, axis=1) / np.sum(test_nums)
    worst_client_acc = np.min(client_avg_accs)
    client_std = np.std(client_avg_accs)

    ood_acc_sum = 0
    ood_weight_sum = 0
    for model_id in range(len(clients)):
        for test_id in range(len(clients)):
            if model_id == test_id:
                continue
            ood_acc_sum += acc_matrix[model_id, test_id] * test_nums[test_id]
            ood_weight_sum += test_nums[test_id]
    ood_acc = ood_acc_sum / ood_weight_sum if ood_weight_sum > 0 else 0
    return round(ind_acc, 4), round(ood_acc, 4), round(worst_client_acc, 4), round(client_std, 4)

def dump_result_record(record, f):
    f.write('{\n')
    items = list(record.items())
    for idx, (key, value) in enumerate(items):
        comma = ',' if idx < len(items) - 1 else ''
        if key == 'acc':
            f.write(f'  "{key}": [\n')
            for row_idx, row in enumerate(value):
                row_comma = ',' if row_idx < len(value) - 1 else ''
                f.write(f'    {json.dumps(row)}{row_comma}\n')
            f.write(f'  ]{comma}\n')
        else:
            f.write(f'  "{key}": {json.dumps(value)}{comma}\n')
    f.write('}\n')

def run(args):
    # initialize server
    server = Server(args)

    # set dataset
    dataset = globals()[args.dataset]

    # initialize clients
    # client image encoder is the same as the global image encoder
    clients = []
    cls_heads = []
    for id, data_name in enumerate(dataset):
        init_image_encoder = copy.deepcopy(server.image_encoder)
        cd = get_data(data_name, server.train_preprocess, server.val_preprocess, args.batch_size, args.num_workers)
        cd = build_subset(cd, args.subset_size)
        cd = split_train_and_val(cd)
        cls_head = server.generate_cls_head(cd, data_name)
        client = Client(args, id, cd.train_dataset, cd.test_dataset, cd.val_dataset, cd.train_loader, cd.test_loader, cd.val_loader, cd.classnames, init_image_encoder, cls_head, data_name)
        clients.append(client)
        cls_heads.append(cls_head)
        del cd

    # generate global cls head
    server.generate_global_cls_head(cls_heads)

    # print("clients[0].model.keys():", clients[0].model.state_dict().keys())
    # print("name of the parameters in clients[0].model:", [k for k,_ in clients[0].model.named_parameters()])
    print("the parameters that require grad in clients[0].model:", [k for k,p in clients[0].model.named_parameters() if p.requires_grad]) # make sure only fine tune the local adapter

    # train and test clients
    total_test_time, total_train_time = 0, 0

    # fine tune clients
    for id in range(len(clients)):
        clients[id].fine_tune(global_round=0)

    start_time = time.time()
    clients, server = proto_initialization(clients, server, args=args)
    train_time = time.time() - start_time
    total_train_time += train_time
    print(f'train time cost: {train_time:.2f}s')

    start_time = time.time()
    for id in range(len(clients)):
        clients[id].fine_tune(global_round=1)
    local_train_time = time.time() - start_time
    total_train_time += local_train_time
    print(f'local adaptation time cost: {local_train_time:.2f}s')

    # cal val loss
    val_loss = 0
    for id in range(len(clients)):
        val_loss += clients[id].cal_val_loss()
    print(f'val loss: {val_loss:.4f}')

    start_time = time.time()
    client_acc = []
    for id, client in enumerate(clients):
        accs = client.test_on_all_clients(clients)
        client_acc.append(accs)
    ind_acc, ood_acc, worst_client_acc, client_std = calculate_summary_acc(client_acc, clients)
    print(f'ind acc: {ind_acc:.4f}, ood acc: {ood_acc:.4f}, worst client acc: {worst_client_acc:.4f}, client std: {client_std:.4f}')

    test_time = time.time() - start_time
    print(f'test time cost: {test_time:.2f}s')
    total_test_time += test_time
    with open(f'./results/ours/{args.image_encoder_name}_{args.dataset}_sub{args.subset_size}_sra{args.sample_ratio}_sram{args.sample_ratio_method}.json', 'a+') as f:
        dump_result_record({'round':0, 'acc': client_acc, 'ind_acc': ind_acc, 'ood_acc': ood_acc, 'worst_client_acc': worst_client_acc, 'client_std': client_std, 'total_test_time': total_test_time, 'total_train_time': total_train_time}, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DomainFL')
    parser.add_argument('-d','--dataset', type=str, default='domainnet', help='Dataset name')
    parser.add_argument('-ss','--subset_size', type=int, default=100, help='Subset size')
    parser.add_argument('-m','--model', type=str, default='CLIP', help='Model name')
    parser.add_argument('-ien','--image_encoder_name', type=str, default='ViT-B-32', help='Image encoder name')
    parser.add_argument('-optim','--optimizer', type=str, default='AdamW', help='Optimizer name')
    parser.add_argument('-lr','--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('-clip','--clip', type=float, default=1, help='Gradient clip')
    parser.add_argument('-bs','--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('-le','--local_epochs', type=int, default=1, help='Number of epochs')
    parser.add_argument('-warm_up','--warm_up', type=int, default=10, help='Warm up epochs')
    parser.add_argument('-gr','--global_rounds', type=int, default=200, help='Number of global rounds')
    parser.add_argument('-device','--device', type=str, default='cuda', help='Device')
    parser.add_argument('-num_workers','--num_workers', type=int, default=12, help='Number of workers')
    parser.add_argument('-eval','--eval_interval', type=int, default=200, help='Log interval')
    parser.add_argument('-did','--device_id', type=str, default=0, help='Device ID')
    parser.add_argument('-seed','--seed', type=int, default=1, help='Seed')
    parser.add_argument('-rw','--regularization_weight', type=float, default=0, help='Regularization weight')
    parser.add_argument('-kdw','--kd_loss_weight', type=float, default=0, help='KD loss weight')
    parser.add_argument('-sra','--sample_ratio', type=float, default=0.1, help='Sample ratio of all embeddings')
    parser.add_argument('-sram','--sample_ratio_method', type=str, default='cluster', help='Sample ratio method (random or cluster, mixed)')
    parser.add_argument('-dp','--diff_privacy', type=float, default=0, help='Diff privacy scale')

    args = parser.parse_args()

    if args.device == 'cuda':
        args.device = torch.device(f'cuda:{args.device_id}' if torch.cuda.is_available() else 'cpu')
    else:
        args.device = torch.device('cpu')

    os.makedirs(f'./results/ours/', exist_ok=True)
    os.makedirs(f'./results/debug/', exist_ok=True)
    with open(f'./results/ours/{args.image_encoder_name}_{args.dataset}_sub{args.subset_size}_sra{args.sample_ratio}_sram{args.sample_ratio_method}.json', 'w+') as f:
        json.dump(generate_json_config(args), f)
        f.write('\n')

    run(args)
