import os

from tqdm import tqdm

import random


def split_train_val_test_set_4_source(graph_path_dict):
    train_set_num = int(len(graph_path_dict) * 0.8)
    data = []
    for gid, graph_path_list in graph_path_dict.items():
        data.append((gid, graph_path_list))
    random.shuffle(data)
    train_set_paths = []
    val_set_paths = []
    for i in range(len(data)):
        gid, g_path_list = data[i]
        if i < train_set_num:
            train_set_paths.extend([(gid, gpath) for gpath in g_path_list])
        else:
            val_set_paths.extend([(gid, gpath) for gpath in g_path_list])
    return train_set_paths, val_set_paths


def split_train_val_test_set_4_target(train_ratio_fine_tuning_list, graph_path_dict):
    max_train_ratio = max(train_ratio_fine_tuning_list)
    test_ratio = (1 - max_train_ratio) / 2
    test_set_num = int(len(graph_path_dict) * test_ratio)
    val_set_num = int(len(graph_path_dict) * (1 - max_train_ratio - test_ratio))
    data = []
    for gid, graph_path_list in graph_path_dict.items():
        data.append((gid, graph_path_list))
    random.shuffle(data)

    train_set_paths_pool = []
    val_set_paths = []
    test_set_paths = []
    for i in range(len(data)):
        gid, g_path_list = data[i]
        if i < test_set_num:
            test_set_paths.extend([(gid, gpath) for gpath in g_path_list])
        elif test_set_num <= i < test_set_num + val_set_num:
            val_set_paths.extend([(gid, gpath) for gpath in g_path_list])
        else:
            train_set_paths_pool.append(data[i])

    paths_with_increasing_ratio = []
    train_ratio_fine_tuning_list.sort()  # increasing order
    current_train_pool = train_set_paths_pool
    last_selected_train = []
    sampled_gid_num = 0
    for train_ratio_fine_tuning in train_ratio_fine_tuning_list:
        train_set_num = int(len(graph_path_dict) * train_ratio_fine_tuning)  # current total gid number
        train_set_paths = []
        train_set_paths.extend(last_selected_train)
        gid_num_to_add = train_set_num - sampled_gid_num
        for i in range(len(current_train_pool)):
            gid, g_path_list = current_train_pool[i]
            if i < gid_num_to_add:
                train_set_paths.extend([(gid, gpath) for gpath in g_path_list])
        # remove selected gid graphs
        current_train_pool = current_train_pool[gid_num_to_add:]
        last_selected_train = train_set_paths
        sampled_gid_num += gid_num_to_add
        # record
        paths_with_increasing_ratio.append((train_set_paths, val_set_paths, test_set_paths))
    return paths_with_increasing_ratio


def write_train_val_test_set_paths(train_set_paths, val_set_paths, test_set_paths, output_path):
    with open(output_path, 'w', encoding='utf8') as file:
        file.write("train set paths:\n")
        for g_id, p in train_set_paths:
            file.write(str(g_id) + " : " + p + "\n")
        file.write("validation set paths:\n")
        for g_id, p in val_set_paths:
            file.write(str(g_id) + " : " + p + "\n")
        file.write("test set paths:\n")
        for g_id, p in test_set_paths:
            file.write(str(g_id) + " : " + p + "\n")


def read_graph_path_dict(directory):
    graph_dataset_dict = {}
    for file in tqdm([f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]):
        gid = file.split("-")[0]
        if gid in graph_dataset_dict:
            graph_dataset_dict[gid].append(os.path.join(directory, file))
        else:
            graph_dataset_dict[gid] = [os.path.join(directory, file)]
    return graph_dataset_dict


def split_data_set_4_source(dataset_name, dataset_id):
    dataset_dir = f"./{dataset_name}_dataset"
    root = f"./{dataset_name}_data/{dataset_id}"
    os.makedirs(root, exist_ok=True)
    specific_train_val_test_set_path = root + f"/specific_dataset_source_{dataset_id}.txt"
    graph_path_dict = read_graph_path_dict(dataset_dir)
    train_set_paths, val_set_paths = split_train_val_test_set_4_source(graph_path_dict)
    write_train_val_test_set_paths(train_set_paths, val_set_paths, [], specific_train_val_test_set_path)


def split_data_set_4_target(dataset_name, dataset_id):
    dataset_dir = f"./{dataset_name}_dataset"
    root = f"./{dataset_name}_data/{dataset_id}"
    os.makedirs(root, exist_ok=True)
    graph_path_dict = read_graph_path_dict(dataset_dir)
    train_ratio_finetuning_list = [0.2, 0.3, 0.4, 0.5, 0.6]
    train_ratio_finetuning_list.sort()
    paths_list = split_train_val_test_set_4_target(train_ratio_finetuning_list, graph_path_dict)
    for idx, (train_set_paths, val_set_paths, test_set_paths) in enumerate(paths_list):
        specific_train_val_test_set_path = root + f"/specific_dataset_target_{dataset_id}_{train_ratio_finetuning_list[idx]}.txt"
        write_train_val_test_set_paths(train_set_paths, val_set_paths, test_set_paths,
                                       specific_train_val_test_set_path)


if __name__ == "__main__":
    dataset_id = 51
    for sys_type in ['source', 'target']:
        for dataset in ['forum', 'halo', 'novel']:
            if sys_type == 'target':
                split_data_set_4_target(dataset_name=dataset, dataset_id=dataset_id)
            else:
                split_data_set_4_source(dataset_name=dataset, dataset_id=dataset_id)
