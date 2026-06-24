import json
import random
from collections import defaultdict

import networkx as nx

import torch
import numpy as np
from torch_geometric.data import Data
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader


def balanced_select(data, n, shuffle_within_gid=True, seed=42):
    """
    data: List[(gid, gpath)]
    n: 保留的总数量
    """
    if seed is not None:
        random.seed(seed)

    # 1. 按 gid 分组
    gid2items = defaultdict(list)
    for gid, gpath in data:
        gid2items[gid].append((gid, gpath))

    # 2. 可选：打乱每个 gid 内部顺序（避免路径偏置）
    if shuffle_within_gid:
        for items in gid2items.values():
            random.shuffle(items)

    # 3. 轮询选取
    selected = []
    gid_list = list(gid2items.keys())

    while len(selected) < n and gid_list:
        new_gid_list = []
        for gid in gid_list:
            if gid2items[gid]:
                selected.append(gid2items[gid].pop())
                if len(selected) == n:
                    break
                # 如果这个 gid 还有剩，保留到下一轮
                if gid2items[gid]:
                    new_gid_list.append(gid)
        gid_list = new_gid_list

    return selected


def construct_graph_to_nx_with_feature(file_path, all_sys_event_cate_dict, all_sys_event_cate_embedding_dict,
                                       exception_embedding_map, filename_embedding_map, gid, label_sys):
    anomalous_nodes = []  # anomalous nodes
    error_type = 'no error'
    edges_with_props = []
    with open(file_path, "r") as file:
        lines = file.readlines()
        start_filter = lines.index("network[son<-parent]=\n")
        for line in lines[:start_filter]:
            if line.startswith("Error_Type="):
                error_type = line.strip().split("=")[1]
            if line.startswith("traceID="):
                trace_id = line.strip().split("=")[1]
            if line.startswith("label="):
                ano_node = line.strip().split("=")[1]
                anomalous_nodes.append(ano_node)

        for line in lines[start_filter + 1:]:
            line = line.strip()
            if line:
                parts = line.split(",")
                edge = parts[0].split("<-")
                edge_info = [edge, parts[3], parts[4], parts[5],
                             1 if edge[0] in anomalous_nodes else 0]  # edge, cost, event, exception
                edges_with_props.append(edge_info)

    G = nx.Graph()
    feature_dim_sys_agnostic = []
    feature_dim_sys_specific = []

    for edge_info in edges_with_props:
        target, source = edge_info[0]
        cost = edge_info[1]
        event = edge_info[2]
        exception = edge_info[3]
        label = edge_info[4]

        if not G.has_edge(target, source):
            G.add_edge(target, source)
        target_feature_temp = (
                np.array(all_sys_event_cate_embedding_dict[event.replace("event=", "", 1)])
                + np.array(filename_embedding_map[target])
                + np.array(exception_embedding_map[exception.replace("exception=", "", 1)])
        ).tolist()
        # target_feature_temp = all_sys_event_cate_embedding_dict[event.replace("event=", "", 1)] + filename_embedding_map[
        #     target] + exception_embedding_map[exception.replace("exception=", "", 1)] + [float(cost.split("=")[1].split("m")[0])]
        target_feature_cate = all_sys_event_cate_embedding_dict[all_sys_event_cate_dict[event.replace("event=", "", 1)]]
        G.nodes[target]["abs"] = all_sys_event_cate_dict[event.replace("event=", "", 1)]  # node event abstraction type

        if not feature_dim_sys_specific:
            feature_dim_sys_specific.append(len(target_feature_temp))
        if not feature_dim_sys_agnostic:
            feature_dim_sys_agnostic.append(len(target_feature_cate))
        G.nodes[target]["label_class"] = label
        G.nodes[target]["gid"] = gid
        G.nodes[target]["label_sys"] = label_sys
        if label == 1:
            G.nodes[target]["error_type"] = error_type
        else:
            G.nodes[target]["error_type"] = 'no error'
        G.nodes[target]["name"] = target
        if G.nodes[target].get('exception', None) is None:
            if exception.replace("exception=", "", 1) != "null":
                G.nodes[target]["exception"] = True
            else:
                G.nodes[target]["exception"] = False
        else:
            if exception.replace("exception=", "", 1) != "null":
                G.nodes[target]["exception"] = True

        feature_value_temp = G.nodes[target].get('feature_temp', None)
        feature_value_cate = G.nodes[target].get('feature_cate', None)
        if feature_value_temp is not None:
            feature_value_temp = [x + y for x, y in zip(target_feature_temp, feature_value_temp)]  # feature add
            G.nodes[target]["feature_temp"] = feature_value_temp
            feature_value_cate = [x + y for x, y in zip(target_feature_cate, feature_value_cate)]  # feature add
            G.nodes[target]["feature_cate"] = feature_value_cate
        else:
            G.nodes[target]["feature_temp"] = target_feature_temp
            G.nodes[target]["feature_cate"] = target_feature_cate
    root_path = []
    for i, node in enumerate(list(G.nodes())):
        if node != "root":
            G.nodes[node]["call_paths"] = [i]
            root_path.append(i)
    """add root node attributes"""
    G.nodes["root"]["feature_temp"] = [float(0)] * feature_dim_sys_specific[0]
    G.nodes["root"]["feature_cate"] = [float(0)] * feature_dim_sys_agnostic[0]
    G.nodes["root"]["label_class"] = 0
    G.nodes["root"]["label_sys"] = label_sys
    G.nodes["root"]["exception"] = False
    G.nodes["root"]["name"] = "root"
    G.nodes["root"]["call_paths"] = root_path
    G.nodes["root"]["error_type"] = 'no error'
    G.nodes["root"]["gid"] = gid
    G.nodes["root"]["abs"] = "null"
    return G


# ==============================
# Convert nx graph to PyG Data
# ==============================
def convert_nx_to_pyg(graph, abs_type_to_id):
    node_list = list(graph.nodes())
    node_index = {n: i for i, n in enumerate(node_list)}

    feats_temp = torch.tensor([graph.nodes[n]["feature_temp"] for n in node_list], dtype=torch.float)
    feats_cate = torch.tensor([graph.nodes[n]["feature_cate"] for n in node_list], dtype=torch.float)

    edges = [[node_index[u], node_index[v]] for u, v in graph.edges]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    labels_class = torch.tensor([graph.nodes[n]["label_class"] for n in node_list], dtype=torch.long)
    labels_sys = torch.tensor([graph.nodes[n]["label_sys"] for n in node_list], dtype=torch.long)
    abs_ids = torch.tensor([abs_type_to_id.get(graph.nodes[n]["abs"], 0) for n in node_list], dtype=torch.long)

    return Data(
        x_temp=feats_temp,
        x_cate=feats_cate,
        edge_index=edge_index,
        y_class=labels_class,
        y_sys=labels_sys,
        abs=abs_ids,
        num_nodes=labels_class.size(0)
    )


# ==============================
# Lazy Dataset
# ==============================
class GraphDataset(Dataset):
    def __init__(self, path_list, all_sys_event_cate_dict, all_sys_event_cate_embedding_dict,
                 exception_embedding_map, filename_embedding_map, abs_type_to_id, label_sys):
        self.path_list = path_list
        self.all_sys_event_cate_dict = all_sys_event_cate_dict
        self.all_sys_event_cate_embedding_dict = all_sys_event_cate_embedding_dict
        self.exception_embedding_map = exception_embedding_map
        self.filename_embedding_map = filename_embedding_map
        self.abs_type_to_id = abs_type_to_id
        self.label_sys = label_sys

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, idx):
        gid, path = self.path_list[idx]
        graph = construct_graph_to_nx_with_feature(
            path,
            self.all_sys_event_cate_dict,
            self.all_sys_event_cate_embedding_dict,
            self.exception_embedding_map,
            self.filename_embedding_map,
            gid,
            self.label_sys[idx]
        )
        return convert_nx_to_pyg(graph, self.abs_type_to_id)


def load_specific_train_val_test_set(args, dataset_path, all_sys_event_cate_dict, all_sys_event_cate_embedding_dict,
                                     exception_embedding_map, filename_embedding_map, label_sys, ratio=0.5,
                                     sys_type='source'):
    with open(dataset_path, 'r', encoding='utf8') as file:
        lines = file.readlines()
    train_idx = lines.index("train set paths:\n")
    val_idx = lines.index("validation set paths:\n")
    test_idx = lines.index("test set paths:\n")
    train_paths = [line.strip() for line in lines[train_idx + 1:val_idx] if line.strip()]
    val_paths = [line.strip() for line in lines[val_idx + 1:test_idx] if line.strip()]
    test_paths = [line.strip() for line in lines[test_idx + 1:] if line.strip()]

    def sample_graphs_by_gid(paths, ratio):
        # 按 gid 分组
        gid_map = defaultdict(list)
        for line in paths:
            gid, path = line.split(" : ")
            gid_map[gid].append((gid, path))

        sampled_paths = []
        for gid, gid_paths in gid_map.items():
            n_total = len(gid_paths)
            n_sample = max(1, int(n_total * ratio))  # 每个 gid 至少抽一个
            sampled_paths.extend(random.sample(gid_paths, n_sample))
        return sampled_paths

    if sys_type == 'target':  # select specific number of graph for fine-tuning
        train_set_paths = []
        for line in train_paths:
            gid, path = line.split(" : ")
            train_set_paths.append((gid.strip(), path.strip()))
        train_set_paths = balanced_select(train_set_paths,
                                          n=args.num_finetuning)  # balanced selection specific number of graph, considering gids
    else:
        train_set_paths = sample_graphs_by_gid(train_paths, ratio)
    val_set_paths = sample_graphs_by_gid(val_paths, ratio)
    test_set_paths = sample_graphs_by_gid(test_paths, ratio)
    return train_set_paths, val_set_paths, test_set_paths


def load_dataset1(args, source=True):
    '''LOAD DATASET'''
    with open(f"./data/final_towards_target_{args.target_dataset}_event_abstraction_c{args.abs_coupling}.json", "r", encoding="utf-8") as f:
        all_sys_event_cate_dict = json.load(f)
    with open(f"./data/unified_level_event_abstraction_embedding_c{args.abs_coupling}.json", "r", encoding="utf-8") as f:
        all_sys_event_cate_embedding_dict = json.load(f)
    with open("./data/unified_exception_embedding.json", "r", encoding="utf-8") as f:
        exception_dict = json.load(f)
    with open("./data/unified_filename_embedding.json", "r", encoding="utf-8") as f:
        filename_dict = json.load(f)

    all_values = list(all_sys_event_cate_dict.values())
    abstract_type_list = sorted(set(all_values))
    abs_type_to_id = {v: i for i, v in enumerate(abstract_type_list)}
    if source:
        all_train_paths = []
        all_val_paths = []
        train_label_sys_list = []
        val_label_sys_list = []
        for label_sys, source_ds in enumerate(args.source_dataset):
            dataset_path = f"./{source_ds}_data/{args.dataset_id}/specific_dataset_source_{args.dataset_id}.txt"  # including training validation set
            train_paths, val_paths, _ = load_specific_train_val_test_set(
                args, dataset_path, all_sys_event_cate_dict, all_sys_event_cate_embedding_dict, exception_dict, filename_dict, label_sys,
                ratio=0.2,
                sys_type='source')
            all_train_paths.extend(train_paths)
            all_val_paths.extend(val_paths)
            train_label_sys_list.extend([label_sys] * len(train_paths))
            val_label_sys_list.extend([label_sys] * len(val_paths))
        train_dataset = GraphDataset(
            all_train_paths, all_sys_event_cate_dict, all_sys_event_cate_embedding_dict,
            exception_dict, filename_dict, abs_type_to_id, label_sys=train_label_sys_list
        )
        val_dataset = GraphDataset(
            all_val_paths, all_sys_event_cate_dict, all_sys_event_cate_embedding_dict,
            exception_dict, filename_dict, abs_type_to_id, label_sys=val_label_sys_list
        )
        print(f"dataset stat: train {len(train_dataset)} | val {len(val_dataset)}")

        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=4, pin_memory=True, persistent_workers=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=4, pin_memory=True, persistent_workers=True
        )
        return train_loader, val_loader
    else:  # target system
        if args.num_finetuning != 0:
            dataset_path = f"./{args.target_dataset}_data/{args.dataset_id}/specific_dataset_target_{args.dataset_id}_{args.ratio_finetuning}.txt"  # including finetuning validation and test set
        else:  # zero-shot
            dataset_path = f"./{args.target_dataset}_data/{args.dataset_id}/specific_dataset_target_{args.dataset_id}_{0.05}.txt"

        train_paths, val_paths, test_paths = load_specific_train_val_test_set(
            args, dataset_path, all_sys_event_cate_dict, all_sys_event_cate_embedding_dict,
            exception_dict, filename_dict, label_sys=0, ratio=1.0, sys_type='target'
        )

        train_dataset = GraphDataset(train_paths, all_sys_event_cate_dict, all_sys_event_cate_embedding_dict,
                                     exception_dict, filename_dict, abs_type_to_id, label_sys=[0] * len(train_paths))
        val_dataset = GraphDataset(val_paths, all_sys_event_cate_dict, all_sys_event_cate_embedding_dict,
                                   exception_dict, filename_dict, abs_type_to_id, label_sys=[0] * len(val_paths))
        test_dataset = GraphDataset(test_paths, all_sys_event_cate_dict, all_sys_event_cate_embedding_dict,
                                    exception_dict, filename_dict, abs_type_to_id, label_sys=[0] * len(test_paths))
        print(f"dataset stat: train {len(train_dataset)} | val {len(val_dataset)} | test {len(test_dataset)}")

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                  num_workers=4, pin_memory=True, persistent_workers=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True,
                                num_workers=4, pin_memory=True, persistent_workers=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True,
                                 num_workers=4, pin_memory=True, persistent_workers=True)

        return train_loader, val_loader, test_loader
