import os

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import pandas as pd


class CustomizedDataset(Dataset):
    def __init__(self, indices_list: list):
        """
        Customized dataset.
        :param indices_list: list, list of indices
        """
        super(CustomizedDataset, self).__init__()

        self.indices_list = indices_list

    def __getitem__(self, idx: int):
        """
        get item at the index in self.indices_list
        :param idx: int, the index
        :return:
        """
        return self.indices_list[idx]

    def __len__(self):
        return len(self.indices_list)


def get_idx_data_loader(indices_list: list, batch_size: int, shuffle: bool):
    """
    get data loader that iterates over indices
    :param indices_list: list, list of indices
    :param batch_size: int, batch size
    :param shuffle: boolean, whether to shuffle the data
    :return: data_loader, DataLoader
    """
    dataset = CustomizedDataset(indices_list=indices_list)

    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             drop_last=False)
    return data_loader


class Data:

    def __init__(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray, node_interact_times: np.ndarray, edge_ids: np.ndarray, labels: np.ndarray):
        """
        Data object to store the nodes interaction information.
        :param src_node_ids: ndarray
        :param dst_node_ids: ndarray
        :param node_interact_times: ndarray
        :param edge_ids: ndarray
        :param labels: ndarray
        """
        self.src_node_ids = src_node_ids
        self.dst_node_ids = dst_node_ids
        self.node_interact_times = node_interact_times
        self.edge_ids = edge_ids
        self.labels = labels
        self.num_interactions = len(src_node_ids)
        self.unique_node_ids = set(src_node_ids) | set(dst_node_ids)
        self.num_unique_nodes = len(self.unique_node_ids)


def get_link_prediction_data(dataset_name: str, val_ratio: float, test_ratio: float):
    """
    generate data for link prediction task (inductive & transductive settings)
    :param dataset_name: str, dataset name
    :param val_ratio: float, validation data ratio
    :param test_ratio: float, test data ratio
    :return: node_raw_features, edge_raw_features, (np.ndarray),
            full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data, (Data object)
    """
    # Load data and train val test split
    graph_df = pd.read_csv('./processed_data/{}/ml_{}.csv'.format(dataset_name, dataset_name))
    edge_raw_features = np.load('./processed_data/{}/ml_{}.npy'.format(dataset_name, dataset_name))
    node_raw_features = np.load('./processed_data/{}/ml_{}_node.npy'.format(dataset_name, dataset_name))

    NODE_FEAT_DIM = EDGE_FEAT_DIM = 172
    NODE_FEAT_DIM = node_raw_features.shape[1]

    assert NODE_FEAT_DIM >= node_raw_features.shape[1], f'Node feature dimension in dataset {dataset_name} is bigger than {NODE_FEAT_DIM}!'
    assert EDGE_FEAT_DIM >= edge_raw_features.shape[1], f'Edge feature dimension in dataset {dataset_name} is bigger than {EDGE_FEAT_DIM}!'
    # padding the features of edges and nodes to the same dimension (172 for all the datasets)
    if node_raw_features.shape[1] < NODE_FEAT_DIM:
        node_zero_padding = np.zeros((node_raw_features.shape[0], NODE_FEAT_DIM - node_raw_features.shape[1]))
        node_raw_features = np.concatenate([node_raw_features, node_zero_padding], axis=1)
    if edge_raw_features.shape[1] < EDGE_FEAT_DIM:
        edge_zero_padding = np.zeros((edge_raw_features.shape[0], EDGE_FEAT_DIM - edge_raw_features.shape[1]))
        edge_raw_features = np.concatenate([edge_raw_features, edge_zero_padding], axis=1)

    assert NODE_FEAT_DIM == node_raw_features.shape[1] and EDGE_FEAT_DIM == edge_raw_features.shape[1], 'Unaligned feature dimensions after feature padding!'

    # get the timestamp of validate and test set
    val_time, test_time = list(np.quantile(graph_df.ts, [(1 - val_ratio - test_ratio), (1 - test_ratio)]))

    src_node_ids = graph_df.u.values.astype(np.longlong)
    dst_node_ids = graph_df.i.values.astype(np.longlong)
    node_interact_times = graph_df.ts.values.astype(np.float64)
    edge_ids = graph_df.idx.values.astype(np.longlong)
    labels = graph_df.label.values

    full_data = Data(src_node_ids=src_node_ids, dst_node_ids=dst_node_ids, node_interact_times=node_interact_times, edge_ids=edge_ids, labels=labels)

    # the setting of seed follows previous works
    random.seed(2020)

    # union to get node set
    node_set = set(src_node_ids) | set(dst_node_ids)
    num_total_unique_node_ids = len(node_set)

    # compute nodes which appear at test time
    test_node_set = set(src_node_ids[node_interact_times > val_time]).union(set(dst_node_ids[node_interact_times > val_time]))
    # sample nodes which we keep as new nodes (to test inductiveness), so then we have to remove all their edges from training
    new_test_node_set = set(random.sample(test_node_set, int(0.1 * num_total_unique_node_ids)))

    # mask for each source and destination to denote whether they are new test nodes
    new_test_source_mask = graph_df.u.map(lambda x: x in new_test_node_set).values
    new_test_destination_mask = graph_df.i.map(lambda x: x in new_test_node_set).values

    # mask, which is true for edges with both destination and source not being new test nodes (because we want to remove all edges involving any new test node)
    observed_edges_mask = np.logical_and(~new_test_source_mask, ~new_test_destination_mask)

    # for train data, we keep edges happening before the validation time which do not involve any new node, used for inductiveness
    train_mask = np.logical_and(node_interact_times <= val_time, observed_edges_mask)

    train_data = Data(src_node_ids=src_node_ids[train_mask], dst_node_ids=dst_node_ids[train_mask],
                      node_interact_times=node_interact_times[train_mask],
                      edge_ids=edge_ids[train_mask], labels=labels[train_mask])

    # define the new nodes sets for testing inductiveness of the model
    train_node_set = set(train_data.src_node_ids).union(train_data.dst_node_ids)
    assert len(train_node_set & new_test_node_set) == 0
    # new nodes that are not in the training set
    new_node_set = node_set - train_node_set

    val_mask = np.logical_and(node_interact_times <= test_time, node_interact_times > val_time)
    test_mask = node_interact_times > test_time

    # new edges with new nodes in the val and test set (for inductive evaluation)
    edge_contains_new_node_mask = np.array([(src_node_id in new_node_set or dst_node_id in new_node_set)
                                            for src_node_id, dst_node_id in zip(src_node_ids, dst_node_ids)])
    new_node_val_mask = np.logical_and(val_mask, edge_contains_new_node_mask)
    new_node_test_mask = np.logical_and(test_mask, edge_contains_new_node_mask)

    # validation and test data
    val_data = Data(src_node_ids=src_node_ids[val_mask], dst_node_ids=dst_node_ids[val_mask],
                    node_interact_times=node_interact_times[val_mask], edge_ids=edge_ids[val_mask], labels=labels[val_mask])

    test_data = Data(src_node_ids=src_node_ids[test_mask], dst_node_ids=dst_node_ids[test_mask],
                     node_interact_times=node_interact_times[test_mask], edge_ids=edge_ids[test_mask], labels=labels[test_mask])

    # validation and test with edges that at least has one new node (not in training set)
    new_node_val_data = Data(src_node_ids=src_node_ids[new_node_val_mask], dst_node_ids=dst_node_ids[new_node_val_mask],
                             node_interact_times=node_interact_times[new_node_val_mask],
                             edge_ids=edge_ids[new_node_val_mask], labels=labels[new_node_val_mask])

    new_node_test_data = Data(src_node_ids=src_node_ids[new_node_test_mask], dst_node_ids=dst_node_ids[new_node_test_mask],
                              node_interact_times=node_interact_times[new_node_test_mask],
                              edge_ids=edge_ids[new_node_test_mask], labels=labels[new_node_test_mask])

    print("The dataset has {} interactions, involving {} different nodes".format(full_data.num_interactions, full_data.num_unique_nodes))
    print("The training dataset has {} interactions, involving {} different nodes".format(
        train_data.num_interactions, train_data.num_unique_nodes))
    print("The validation dataset has {} interactions, involving {} different nodes".format(
        val_data.num_interactions, val_data.num_unique_nodes))
    print("The test dataset has {} interactions, involving {} different nodes".format(
        test_data.num_interactions, test_data.num_unique_nodes))
    print("The new node validation dataset has {} interactions, involving {} different nodes".format(
        new_node_val_data.num_interactions, new_node_val_data.num_unique_nodes))
    print("The new node test dataset has {} interactions, involving {} different nodes".format(
        new_node_test_data.num_interactions, new_node_test_data.num_unique_nodes))
    print("{} nodes were used for the inductive testing, i.e. are never seen during training".format(len(new_test_node_set)))

    return node_raw_features, edge_raw_features, full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data


def get_few_shot_link_prediction_data(dataset_name: str, pretrain_ratio: float, finetune_size: int):
    """
    Generate data for link prediction task (pretrain, finetune, validation, test) with transductive and strict inductive settings.
    :param dataset_name: str, dataset name.
    :param pretrain_ratio: float, pretrain data ratio based on time (e.g. 0.8 for 80% of time-based data as pretrain).
    :param finetune_size: int, the number of edges used for finetune and validation (e.g. 500-shot finetuning).
    :return: node_raw_features, edge_raw_features, (np.ndarray),
             full_data, pretrain_data, finetune_data, val_data, test_data, new_node_val_data, new_node_test_data, (Data object).
    """
    # Load data and split for train, val, and test
    graph_df = pd.read_csv(f'./processed_data/{dataset_name}/ml_{dataset_name}.csv')
    edge_raw_features = np.load(f'./processed_data/{dataset_name}/ml_{dataset_name}.npy')
    node_raw_features = np.load(f'./processed_data/{dataset_name}/ml_{dataset_name}_node.npy')

    NODE_FEAT_DIM = EDGE_FEAT_DIM = 172
    NODE_FEAT_DIM = node_raw_features.shape[1]

    assert NODE_FEAT_DIM >= node_raw_features.shape[
        1], f'Node feature dimension in dataset {dataset_name} is bigger than {NODE_FEAT_DIM}!'
    assert EDGE_FEAT_DIM >= edge_raw_features.shape[
        1], f'Edge feature dimension in dataset {dataset_name} is bigger than {EDGE_FEAT_DIM}!'

    # Padding the features of edges and nodes to the same dimension (172 for all the datasets)
    if node_raw_features.shape[1] < NODE_FEAT_DIM:
        node_zero_padding = np.zeros((node_raw_features.shape[0], NODE_FEAT_DIM - node_raw_features.shape[1]))
        node_raw_features = np.concatenate([node_raw_features, node_zero_padding], axis=1)
    if edge_raw_features.shape[1] < EDGE_FEAT_DIM:
        edge_zero_padding = np.zeros((edge_raw_features.shape[0], EDGE_FEAT_DIM - edge_raw_features.shape[1]))
        edge_raw_features = np.concatenate([edge_raw_features, edge_zero_padding], axis=1)

    assert NODE_FEAT_DIM == node_raw_features.shape[1] and EDGE_FEAT_DIM == edge_raw_features.shape[
        1], 'Unaligned feature dimensions after feature padding!'

    # Step 1: Time-based splitting for pretraining data
    pretrain_time_threshold = np.quantile(graph_df.ts, pretrain_ratio)

    # Split data based on time threshold for pretrain set
    pretrain_mask = graph_df.ts <= pretrain_time_threshold
    pretrain_df = graph_df[pretrain_mask]
    remaining_data = graph_df[~pretrain_mask]

    # Step 2: Edge-based splitting for finetune, validation, and test data
    finetune_df = remaining_data.iloc[:finetune_size]
    val_df = remaining_data.iloc[finetune_size: 2 * finetune_size]
    test_df = remaining_data.iloc[2 * finetune_size:]

    random.seed(2024)

    full_data = Data(
        src_node_ids=graph_df.u.values.astype(np.longlong),
        dst_node_ids=graph_df.i.values.astype(np.longlong),
        node_interact_times=graph_df.ts.values.astype(np.float64),
        edge_ids=graph_df.idx.values.astype(np.longlong),
        labels=graph_df.label.values
    )

    # Step 3: Identify new nodes in the dataset (those that only appear after pretrain set)
    node_set = set(graph_df.u.values).union(set(graph_df.i.values))
    test_node_set = set(test_df.u.values).union(set(test_df.i.values))

    # Sample 10% of nodes in test_node_set to be new test nodes
    new_test_node_set = set(random.sample(test_node_set, int(0.1 * len(node_set))))

    # Create masks for new test nodes in both pretrain and finetune datasets
    new_test_source_mask_pretrain = pretrain_df.u.isin(new_test_node_set).values
    new_test_destination_mask_pretrain = pretrain_df.i.isin(new_test_node_set).values
    new_test_source_mask_finetune = finetune_df.u.isin(new_test_node_set).values
    new_test_destination_mask_finetune = finetune_df.i.isin(new_test_node_set).values

    # Create observed edges mask (true for edges not involving new test nodes)
    observed_edges_mask_pretrain = np.logical_and(~new_test_source_mask_pretrain, ~new_test_destination_mask_pretrain)
    observed_edges_mask_finetune = np.logical_and(~new_test_source_mask_finetune, ~new_test_destination_mask_finetune)

    # Step 4: Filter pretrain and finetune datasets to remove edges involving new test nodes
    pretrain_data = Data(
        src_node_ids=pretrain_df.u.values[observed_edges_mask_pretrain].astype(np.longlong),
        dst_node_ids=pretrain_df.i.values[observed_edges_mask_pretrain].astype(np.longlong),
        node_interact_times=pretrain_df.ts.values[observed_edges_mask_pretrain].astype(np.float64),
        edge_ids=pretrain_df.idx.values[observed_edges_mask_pretrain].astype(np.longlong),
        labels=pretrain_df.label.values[observed_edges_mask_pretrain]
    )

    finetune_data = Data(
        src_node_ids=finetune_df.u.values[observed_edges_mask_finetune].astype(np.longlong),
        dst_node_ids=finetune_df.i.values[observed_edges_mask_finetune].astype(np.longlong),
        node_interact_times=finetune_df.ts.values[observed_edges_mask_finetune].astype(np.float64),
        edge_ids=finetune_df.idx.values[observed_edges_mask_finetune].astype(np.longlong),
        labels=finetune_df.label.values[observed_edges_mask_finetune]
    )

    val_data = Data(
        src_node_ids=val_df.u.values.astype(np.longlong),
        dst_node_ids=val_df.i.values.astype(np.longlong),
        node_interact_times=val_df.ts.values.astype(np.float64),
        edge_ids=val_df.idx.values.astype(np.longlong),
        labels=val_df.label.values
    )

    test_data = Data(
        src_node_ids=test_df.u.values.astype(np.longlong),
        dst_node_ids=test_df.i.values.astype(np.longlong),
        node_interact_times=test_df.ts.values.astype(np.float64),
        edge_ids=test_df.idx.values.astype(np.longlong),
        labels=test_df.label.values
    )

    # Step 5: Inductive validation and test sets - filter edges that involve new nodes
    inductive_val_mask = np.logical_or(val_df.u.isin(new_test_node_set), val_df.i.isin(new_test_node_set))
    inductive_test_mask = np.logical_or(test_df.u.isin(new_test_node_set), test_df.i.isin(new_test_node_set))

    new_node_val_data = Data(
        src_node_ids=val_df.u.values[inductive_val_mask].astype(np.longlong),
        dst_node_ids=val_df.i.values[inductive_val_mask].astype(np.longlong),
        node_interact_times=val_df.ts.values[inductive_val_mask].astype(np.float64),
        edge_ids=val_df.idx.values[inductive_val_mask].astype(np.longlong),
        labels=val_df.label.values[inductive_val_mask]
    )

    new_node_test_data = Data(
        src_node_ids=test_df.u.values[inductive_test_mask].astype(np.longlong),
        dst_node_ids=test_df.i.values[inductive_test_mask].astype(np.longlong),
        node_interact_times=test_df.ts.values[inductive_test_mask].astype(np.float64),
        edge_ids=test_df.idx.values[inductive_test_mask].astype(np.longlong),
        labels=test_df.label.values[inductive_test_mask]
    )

    # Print dataset statistics
    print(f"The dataset has {len(graph_df)} interactions, involving {len(node_set)} different nodes.")
    print(
        f"The pretrain dataset has {pretrain_data.num_interactions} interactions, involving {pretrain_data.num_unique_nodes} different nodes.")
    print(
        f"The finetune dataset has {finetune_data.num_interactions} interactions, involving {finetune_data.num_unique_nodes} different nodes.")
    print(
        f"The validation dataset has {val_data.num_interactions} interactions, involving {val_data.num_unique_nodes} different nodes.")
    print(
        f"The test dataset has {test_data.num_interactions} interactions, involving {test_data.num_unique_nodes} different nodes.")
    print(
        f"The new node validation dataset has {new_node_val_data.num_interactions} interactions, involving {new_node_val_data.num_unique_nodes} different nodes.")
    print(
        f"The new node test dataset has {new_node_test_data.num_interactions} interactions, involving {new_node_test_data.num_unique_nodes} different nodes.")

    return node_raw_features, edge_raw_features, full_data, pretrain_data, finetune_data, val_data, test_data, new_node_val_data, new_node_test_data

def get_node_classification_data(dataset_name: str, val_ratio: float, test_ratio: float):
    """
    generate data for node classification task
    :param dataset_name: str, dataset name
    :param val_ratio: float, validation data ratio
    :param test_ratio: float, test data ratio
    :return: node_raw_features, edge_raw_features, (np.ndarray),
            full_data, train_data, val_data, test_data, (Data object)
    """
    # Load data and train val test split
    graph_df = pd.read_csv('./processed_data/{}/ml_{}.csv'.format(dataset_name, dataset_name))
    edge_raw_features = np.load('./processed_data/{}/ml_{}.npy'.format(dataset_name, dataset_name))
    node_raw_features = np.load('./processed_data/{}/ml_{}_node.npy'.format(dataset_name, dataset_name))


    NODE_FEAT_DIM = EDGE_FEAT_DIM = 172
    NODE_FEAT_DIM = node_raw_features.shape[1]


    assert NODE_FEAT_DIM >= node_raw_features.shape[1], f'Node feature dimension in dataset {dataset_name} is bigger than {NODE_FEAT_DIM}!'
    assert EDGE_FEAT_DIM >= edge_raw_features.shape[1], f'Edge feature dimension in dataset {dataset_name} is bigger than {EDGE_FEAT_DIM}!'
    # padding the features of edges and nodes to the same dimension (172 for all the datasets)
    if node_raw_features.shape[1] < NODE_FEAT_DIM:
        node_zero_padding = np.zeros((node_raw_features.shape[0], NODE_FEAT_DIM - node_raw_features.shape[1]))
        node_raw_features = np.concatenate([node_raw_features, node_zero_padding], axis=1)
    if edge_raw_features.shape[1] < EDGE_FEAT_DIM:
        edge_zero_padding = np.zeros((edge_raw_features.shape[0], EDGE_FEAT_DIM - edge_raw_features.shape[1]))
        edge_raw_features = np.concatenate([edge_raw_features, edge_zero_padding], axis=1)

    assert NODE_FEAT_DIM == node_raw_features.shape[1] and EDGE_FEAT_DIM == edge_raw_features.shape[1], 'Unaligned feature dimensions after feature padding!'

    # get the timestamp of validate and test set
    val_time, test_time = list(np.quantile(graph_df.ts, [(1 - val_ratio - test_ratio), (1 - test_ratio)]))

    src_node_ids = graph_df.u.values.astype(np.longlong)
    dst_node_ids = graph_df.i.values.astype(np.longlong)
    node_interact_times = graph_df.ts.values.astype(np.float64)
    edge_ids = graph_df.idx.values.astype(np.longlong)
    labels = graph_df.label.values

    # The setting of seed follows previous works
    random.seed(2020)

    train_mask = node_interact_times <= val_time
    val_mask = np.logical_and(node_interact_times <= test_time, node_interact_times > val_time)
    test_mask = node_interact_times > test_time

    full_data = Data(src_node_ids=src_node_ids, dst_node_ids=dst_node_ids, node_interact_times=node_interact_times, edge_ids=edge_ids, labels=labels)
    train_data = Data(src_node_ids=src_node_ids[train_mask], dst_node_ids=dst_node_ids[train_mask],
                      node_interact_times=node_interact_times[train_mask],
                      edge_ids=edge_ids[train_mask], labels=labels[train_mask])
    val_data = Data(src_node_ids=src_node_ids[val_mask], dst_node_ids=dst_node_ids[val_mask],
                    node_interact_times=node_interact_times[val_mask], edge_ids=edge_ids[val_mask], labels=labels[val_mask])
    test_data = Data(src_node_ids=src_node_ids[test_mask], dst_node_ids=dst_node_ids[test_mask],
                     node_interact_times=node_interact_times[test_mask], edge_ids=edge_ids[test_mask], labels=labels[test_mask])

    return node_raw_features, edge_raw_features, full_data, train_data, val_data, test_data

def get_few_shot_node_classification_data(dataset_name: str, pretrain_ratio: float, finetune_size: int):
    """
    Generate data for link prediction task (pretrain, finetune, validation, test) with transductive and strict inductive settings.
    :param dataset_name: str, dataset name.
    :param pretrain_ratio: float, pretrain data ratio based on time (e.g. 0.8 for 80% of time-based data as pretrain).
    :param finetune_size: int, the number of edges used for finetune and validation (e.g. 500-shot finetuning).
    :return: node_raw_features, edge_raw_features, (np.ndarray),
             full_data, pretrain_data, finetune_data, val_data, test_data, new_node_val_data, new_node_test_data, (Data object).
    """
    # Load data and split for train, val, and test
    graph_df = pd.read_csv(f'./processed_data/{dataset_name}/ml_{dataset_name}.csv')
    edge_raw_features = np.load(f'./processed_data/{dataset_name}/ml_{dataset_name}.npy')
    node_raw_features = np.load(f'./processed_data/{dataset_name}/ml_{dataset_name}_node.npy')

    NODE_FEAT_DIM = EDGE_FEAT_DIM = 172
    NODE_FEAT_DIM = node_raw_features.shape[1]

    assert NODE_FEAT_DIM >= node_raw_features.shape[
        1], f'Node feature dimension in dataset {dataset_name} is bigger than {NODE_FEAT_DIM}!'
    assert EDGE_FEAT_DIM >= edge_raw_features.shape[
        1], f'Edge feature dimension in dataset {dataset_name} is bigger than {EDGE_FEAT_DIM}!'

    # Padding the features of edges and nodes to the same dimension (172 for all the datasets)
    if node_raw_features.shape[1] < NODE_FEAT_DIM:
        node_zero_padding = np.zeros((node_raw_features.shape[0], NODE_FEAT_DIM - node_raw_features.shape[1]))
        node_raw_features = np.concatenate([node_raw_features, node_zero_padding], axis=1)
    if edge_raw_features.shape[1] < EDGE_FEAT_DIM:
        edge_zero_padding = np.zeros((edge_raw_features.shape[0], EDGE_FEAT_DIM - edge_raw_features.shape[1]))
        edge_raw_features = np.concatenate([edge_raw_features, edge_zero_padding], axis=1)

    assert NODE_FEAT_DIM == node_raw_features.shape[1] and EDGE_FEAT_DIM == edge_raw_features.shape[
        1], 'Unaligned feature dimensions after feature padding!'

    # Step 1: Time-based splitting for pretraining data
    pretrain_time_threshold = np.quantile(graph_df.ts, pretrain_ratio)

    # Split data based on time threshold for pretrain set
    pretrain_mask = graph_df.ts <= pretrain_time_threshold
    pretrain_df = graph_df[pretrain_mask]
    remaining_data = graph_df[~pretrain_mask]


    full_data = Data(
        src_node_ids=graph_df.u.values.astype(np.longlong),
        dst_node_ids=graph_df.i.values.astype(np.longlong),
        node_interact_times=graph_df.ts.values.astype(np.float64),
        edge_ids=graph_df.idx.values.astype(np.longlong),
        labels=graph_df.label.values
    )

    label_one_edge_id = []
    for src, dst, ts, edge_ids, label in zip(remaining_data.u, remaining_data.i, remaining_data.ts, remaining_data.idx, remaining_data.label):
        if label == 1:
            label_one_edge_id.append(edge_ids)


    select_label_one_edge_id = label_one_edge_id[0]  # 保证至少有1个标签为1
    # 其余标签随便
    a = remaining_data.idx.drop(select_label_one_edge_id)
    select_label_other_edge_id = a[:finetune_size - 1]

    label_one_mask = np.isin(remaining_data.idx, select_label_one_edge_id)
    label_other_mask = np.isin(remaining_data.idx, select_label_other_edge_id)
    mask = np.logical_or(label_one_mask, label_other_mask)

    finetune_data = Data(src_node_ids=remaining_data.u[mask].values.astype(np.longlong),
                      dst_node_ids=remaining_data.i[mask].values.astype(np.longlong),
                      node_interact_times=remaining_data.ts[mask].values.astype(np.longlong),
                      edge_ids=remaining_data.idx[mask].values.astype(np.longlong), labels=remaining_data.label[mask].values.astype(np.longlong))


    val_time = finetune_data.node_interact_times[-1]
    val_and_test_mask = graph_df.ts > val_time
    remaining_data = graph_df[val_and_test_mask]

    label_one_edge_id = []
    for src, dst, ts, edge_ids, label in zip(remaining_data.u, remaining_data.i, remaining_data.ts, remaining_data.idx, remaining_data.label):
        if label == 1:
            label_one_edge_id.append(edge_ids)


    select_label_one_edge_id = label_one_edge_id[0]  # 保证至少有1个标签为1
    # 其余标签随便
    a = remaining_data.idx.drop(select_label_one_edge_id)
    select_label_other_edge_id = a[:finetune_size - 1]

    label_one_mask = np.isin(remaining_data.idx, select_label_one_edge_id)
    label_other_mask = np.isin(remaining_data.idx, select_label_other_edge_id)
    mask = np.logical_or(label_one_mask, label_other_mask)

    val_data = Data(src_node_ids=remaining_data.u[mask].values.astype(np.longlong),
                      dst_node_ids=remaining_data.i[mask].values.astype(np.longlong),
                      node_interact_times=remaining_data.ts[mask].values.astype(np.longlong),
                      edge_ids=remaining_data.idx[mask].values.astype(np.longlong), labels=remaining_data.label[mask].values.astype(np.longlong))

    test_time = val_data.node_interact_times[-1]
    test_mask = graph_df.ts > test_time
    test_df = graph_df[test_mask]



    node_set = set(graph_df.u.values).union(set(graph_df.i.values))


    # Step 4: Filter pretrain and finetune datasets to remove edges involving new test nodes
    pretrain_data = Data(
        src_node_ids=pretrain_df.u.values.astype(np.longlong),
        dst_node_ids=pretrain_df.i.values.astype(np.longlong),
        node_interact_times=pretrain_df.ts.values.astype(np.float64),
        edge_ids=pretrain_df.idx.values.astype(np.longlong),
        labels=pretrain_df.label.values
    )

    test_data = Data(
        src_node_ids=test_df.u.values.astype(np.longlong),
        dst_node_ids=test_df.i.values.astype(np.longlong),
        node_interact_times=test_df.ts.values.astype(np.float64),
        edge_ids=test_df.idx.values.astype(np.longlong),
        labels=test_df.label.values
    )



    # Print dataset statistics
    print(f"The dataset has {len(graph_df)} interactions, involving {len(node_set)} different nodes.")
    print(
        f"The pretrain dataset has {pretrain_data.num_interactions} interactions, involving {pretrain_data.num_unique_nodes} different nodes.")
    print(
        f"The finetune dataset has {finetune_data.num_interactions} interactions, involving {finetune_data.num_unique_nodes} different nodes.")
    print(
        f"The validation dataset has {val_data.num_interactions} interactions, involving {val_data.num_unique_nodes} different nodes.")
    print(
        f"The test dataset has {test_data.num_interactions} interactions, involving {test_data.num_unique_nodes} different nodes.")

    return node_raw_features, edge_raw_features, full_data, pretrain_data, finetune_data, val_data, test_data


def split_train_data(train_data, pretrain_ratio=0.5, finetune_ratio=0.2):
    """
    Split the train data into pretrain and finetune datasets based on interaction times.
    :param train_data: Data, the original training dataset.
    :param pretrain_ratio: float, the ratio of data to be used for pretraining.
    :param finetune_ratio: float, the ratio of data to be used for finetuning.
    :return: pretrain_data, finetune_data (Data objects)
    """
    # Calculate the time split for pretrain and finetune datasets
    pretrain_time, finetune_time = list(np.quantile(train_data.node_interact_times, [pretrain_ratio/(pretrain_ratio + finetune_ratio)
        , 1]))

    # Pretrain data: interactions before pretrain_time
    pretrain_mask = train_data.node_interact_times <= pretrain_time
    pretrain_data = Data(src_node_ids=train_data.src_node_ids[pretrain_mask], dst_node_ids=train_data.dst_node_ids[pretrain_mask],
                         node_interact_times=train_data.node_interact_times[pretrain_mask],
                         edge_ids=train_data.edge_ids[pretrain_mask], labels=train_data.labels[pretrain_mask])

    # Finetune data: interactions between pretrain_time and finetune_time
    finetune_mask = np.logical_and(train_data.node_interact_times > pretrain_time, train_data.node_interact_times <= finetune_time)
    finetune_data = Data(src_node_ids=train_data.src_node_ids[finetune_mask], dst_node_ids=train_data.dst_node_ids[finetune_mask],
                         node_interact_times=train_data.node_interact_times[finetune_mask],
                         edge_ids=train_data.edge_ids[finetune_mask], labels=train_data.labels[finetune_mask])

    return pretrain_data, finetune_data

# Save the split datasets
def save_data(data, save_split_data_folder, filename):
    save_path = os.path.join(save_split_data_folder, f"{filename}.npz")
    np.savez(save_path,
             src_node_ids=data.src_node_ids,
             dst_node_ids=data.dst_node_ids,
             node_interact_times=data.node_interact_times,
             edge_ids=data.edge_ids,
             labels=data.labels)

def load_data(save_split_data_folder, filename):
    load_path = os.path.join(save_split_data_folder, f"{filename}.npz")
    data = np.load(load_path)
    return Data(src_node_ids=data['src_node_ids'],
                dst_node_ids=data['dst_node_ids'],
                node_interact_times=data['node_interact_times'],
                edge_ids=data['edge_ids'],
                labels=data['labels'])


def save_all_spilt_data(save_split_data_folder,pretrain_data,finetune_data,val_data,test_data,new_node_val_data=None,new_node_test_data=None):
    save_data(pretrain_data, save_split_data_folder, 'pretrain_data')
    save_data(finetune_data, save_split_data_folder, 'finetune_data')
    save_data(val_data, save_split_data_folder, 'val_data')
    save_data(test_data, save_split_data_folder, 'test_data')
    if new_node_val_data:
        save_data(new_node_val_data, save_split_data_folder, 'new_node_val_data')
    if new_node_test_data:
        save_data(new_node_test_data, save_split_data_folder, 'new_node_test_data')

def load_all_spilt_data(load_split_data_folder):
    pretrain_data = load_data(load_split_data_folder, 'pretrain_data')
    finetune_data = load_data(load_split_data_folder, 'finetune_data')
    val_data = load_data(load_split_data_folder, 'val_data')
    test_data = load_data(load_split_data_folder, 'test_data')
    new_node_val_data = load_data(load_split_data_folder, 'new_node_val_data')
    new_node_test_data = load_data(load_split_data_folder, 'new_node_test_data')
    return pretrain_data, finetune_data, val_data, test_data, new_node_val_data, new_node_test_data

def generate_few_shot_dataset(dataset,train_shotnum,classnum,tasknum, seed=0,task_type="node"):

    train_datas = []
    degrees = dict()
    node2hist = dict()
    label_one_edge_id = []


    for src, dst, edge_ids, label in zip(dataset.src_node_ids, dataset.dst_node_ids, dataset.edge_ids, dataset.labels):

        if src not in degrees:
            degrees[src] = 0
        if dst not in degrees:
            degrees[dst] = 0

        if src not in node2hist:
            node2hist[src] = list()

        if dst not in node2hist:
            node2hist[dst] = list()

        if label == 1:
            label_one_edge_id.append(edge_ids)

        degrees[int(src)] += 1
        degrees[int(dst)] += 1

    filtered_node_ids = []

    for node_id, degree in degrees.items():
        if degree >= 2:
            filtered_node_ids.append(node_id)


    if task_type == "link":
        random.seed(seed)
        np.random.seed(seed)

        # ls = []

        for task in range(5):  # tasknum

            node_unique_ids = filtered_node_ids

            select_node_unique_id = np.random.choice(node_unique_ids, 3, replace=False)     # 样本数
            src_mask = np.isin(dataset.src_node_ids, select_node_unique_id)
            dst_mask = np.isin(dataset.dst_node_ids, select_node_unique_id)
            mask = np.logical_or(src_mask,dst_mask)

            # mask = np.zeros(length, dtype=bool)
            # mask[select_edge_id] = True

            train_data = Data(src_node_ids=dataset.src_node_ids[mask],
                              dst_node_ids=dataset.dst_node_ids[mask],
                              node_interact_times=dataset.node_interact_times[mask],
                              edge_ids=dataset.edge_ids[mask], labels=dataset.labels[mask])
            # l = train_data.src_node_ids.size
            # ls.append(l)
            train_datas.append(train_data)  # 100个30shot正样本


    elif task_type == "node":       # 30条边
        np.random.seed(seed)

        for task in range(5):  # tasknum
            select_label_one_edge_id = np.random.choice(label_one_edge_id, 1, replace=False)  # 保证至少有1个标签为1
            # 其余标签随便
            select_label_other_edge_id = np.random.choice(dataset.edge_ids - select_label_one_edge_id, 30 - 1,
                                                          replace=False)
            label_one_mask = np.isin(dataset.edge_ids, select_label_one_edge_id)
            label_other_mask = np.isin(dataset.edge_ids, select_label_other_edge_id)
            mask = np.logical_or(label_one_mask,label_other_mask)

            train_data = Data(src_node_ids=dataset.src_node_ids[mask],
                              dst_node_ids=dataset.dst_node_ids[mask],
                              node_interact_times=dataset.node_interact_times[mask],
                              edge_ids=dataset.edge_ids[mask], labels=dataset.labels[mask])
            # l = train_data.src_node_ids.size
            # ls.append(l)
            train_datas.append(train_data)  # 100个30shot正样本



    return train_datas


