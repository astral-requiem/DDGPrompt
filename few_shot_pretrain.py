import logging
import time
import sys
import os
from tqdm import tqdm
import numpy as np
import warnings
import shutil
import json
import torch
import torch.nn as nn

from models.DF import DF
from models.DF2 import DF2
from models.TGAT import TGAT
from models.MemoryModel import MemoryModel, compute_src_dst_node_time_shifts
from models.CAWN import CAWN
from models.TCL import TCL
from models.GraphMixer import GraphMixer
from models.DyGFormer import DyGFormer
from models.modules import MergeLayer
from utils.utils import set_random_seed, convert_to_gpu, get_parameter_sizes, create_optimizer, contrastive_loss
from utils.utils import get_neighbor_sampler, NegativeEdgeSampler
from evaluate_models_utils import evaluate_model_link_prediction
from utils.metrics import get_link_prediction_metrics
from utils.DataLoader import get_idx_data_loader, get_link_prediction_data, split_train_data, save_all_spilt_data,get_few_shot_link_prediction_data
from utils.EarlyStopping import EarlyStopping
from utils.load_configs import get_link_prediction_args

class Data:
    def __init__(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray,
                 node_interact_times: np.ndarray, edge_ids: np.ndarray, labels: np.ndarray):
        self.src_node_ids = src_node_ids
        self.dst_node_ids = dst_node_ids
        self.node_interact_times = node_interact_times
        self.edge_ids = edge_ids
        self.labels = labels
        self.num_interactions = len(src_node_ids)
        self.unique_node_ids = set(src_node_ids) | set(dst_node_ids)
        self.num_unique_nodes = len(self.unique_node_ids)

def get_interaction_counts(data: Data) -> dict:
    counts = {}
    for node_id in np.concatenate([data.src_node_ids, data.dst_node_ids]):
        counts[node_id] = counts.get(node_id, 0) + 1
    return counts

def filter_edges_by_interaction(data: Data, interaction_counts: dict, max_interactions: int) -> Data:

    a = np.array([interaction_counts.get(node_id, 0) < max_interactions for node_id in data.src_node_ids])
    b = np.array([interaction_counts.get(node_id, 0) < max_interactions for node_id in data.dst_node_ids])
    mask = np.logical_or(a,b)

    return Data(
        src_node_ids=data.src_node_ids[mask],
        dst_node_ids=data.dst_node_ids[mask],
        node_interact_times=data.node_interact_times[mask],
        edge_ids=data.edge_ids[mask],
        labels=data.labels[mask]
    )

def process_datasets(full_data: Data, pretrain_data: Data, finetune_data: Data,
                     val_data: Data, test_data: Data, new_node_val_data: Data,
                     new_node_test_data: Data, max_interactions: int) -> tuple:
    interaction_counts = get_interaction_counts(full_data)

    filtered_full_data = filter_edges_by_interaction(full_data, interaction_counts, max_interactions)
    filtered_pretrain_data = filter_edges_by_interaction(pretrain_data, interaction_counts, max_interactions)
    filtered_finetune_data = filter_edges_by_interaction(finetune_data, interaction_counts, max_interactions)
    filtered_val_data = filter_edges_by_interaction(val_data, interaction_counts, max_interactions)
    filtered_test_data = filter_edges_by_interaction(test_data, interaction_counts, max_interactions)
    filtered_new_node_val_data = filter_edges_by_interaction(new_node_val_data, interaction_counts, max_interactions)
    filtered_new_node_test_data = filter_edges_by_interaction(new_node_test_data, interaction_counts, max_interactions)

    print(f"Filtered Full Data: {filtered_full_data.num_interactions} interactions")
    print(f"Filtered Pretrain Data: {filtered_pretrain_data.num_interactions} interactions")
    print(f"Filtered Finetune Data: {filtered_finetune_data.num_interactions} interactions")
    print(f"Filtered Validation Data: {filtered_val_data.num_interactions} interactions")
    print(f"Filtered Test Data: {filtered_test_data.num_interactions} interactions")
    print(f"Filtered New Node Validation Data: {filtered_new_node_val_data.num_interactions} interactions")
    print(f"Filtered New Node Test Data: {filtered_new_node_test_data.num_interactions} interactions")

    return (
        filtered_full_data, filtered_pretrain_data, filtered_finetune_data,
        filtered_val_data, filtered_test_data, filtered_new_node_val_data, filtered_new_node_test_data
    )

if __name__ == "__main__":

    warnings.filterwarnings('ignore')

    # get arguments
    args = get_link_prediction_args(is_evaluation=False)

    args.val_ratio = 0.01
    args.test_ratio = 0.18
    args.pretrain_ratio = 0.8
    # args.pretrain_ratio = 0.7
    args.finetune_ratio = 0.01      # 80,1,1,18
    # args.finetune_ratio = 0.10      # 80,1,1,18

    node_raw_features, edge_raw_features, full_data, pretrain_data, finetune_data, val_data, test_data, new_node_val_data, new_node_test_data = \
        get_few_shot_link_prediction_data(dataset_name=args.dataset_name, pretrain_ratio=0.8, finetune_size=70)

    full_data, pretrain_data, finetune_data, val_data, test_data, new_node_val_data, new_node_test_data = \
        process_datasets(full_data, pretrain_data, finetune_data, val_data, test_data, new_node_val_data, new_node_test_data, 100)



    # initialize training neighbor sampler to retrieve temporal graph
    train_neighbor_sampler = get_neighbor_sampler(data=pretrain_data, sample_neighbor_strategy=args.sample_neighbor_strategy,
                                                  time_scaling_factor=args.time_scaling_factor, seed=0)

    # initialize validation and test neighbor sampler to retrieve temporal graph
    full_neighbor_sampler = get_neighbor_sampler(data=full_data, sample_neighbor_strategy=args.sample_neighbor_strategy,
                                                 time_scaling_factor=args.time_scaling_factor, seed=1)

    # initialize negative samplers, set seeds for validation and testing so negatives are the same across different runs
    # in the inductive setting, negatives are sampled only amongst other new nodes
    # train negative edge sampler does not need to specify the seed, but evaluation samplers need to do so
    train_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=pretrain_data.src_node_ids, dst_node_ids=pretrain_data.dst_node_ids)
    val_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=full_data.src_node_ids, dst_node_ids=full_data.dst_node_ids, seed=0)
    new_node_val_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=new_node_val_data.src_node_ids, dst_node_ids=new_node_val_data.dst_node_ids, seed=1)
    test_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=full_data.src_node_ids, dst_node_ids=full_data.dst_node_ids, seed=2)
    new_node_test_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=new_node_test_data.src_node_ids, dst_node_ids=new_node_test_data.dst_node_ids, seed=3)

    # get data loaders
    train_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(pretrain_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)
    val_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(val_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)
    new_node_val_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(new_node_val_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)
    test_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(test_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)
    new_node_test_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(new_node_test_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)


    set_random_seed(seed=2025)

    args.seed = 2048  
    args.save_model_name = f'{args.model_name}_seed{args.seed}'


    # set up logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    os.makedirs(f"./logs/{args.model_name}/{args.dataset_name}/{args.save_model_name}/", exist_ok=True)
    # create file handler that logs debug and higher level messages
    fh = logging.FileHandler(f"./logs/{args.model_name}/{args.dataset_name}/{args.save_model_name}/{str(time.time())}.log")
    fh.setLevel(logging.DEBUG)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to logger
    logger.addHandler(fh)
    logger.addHandler(ch)

    run_start_time = time.time()
    logger.info(f"********** Pre-training starts. **********")

    logger.info(f'configuration is {args}')

    # create model
    if args.model_name == 'TGAT':
        dynamic_backbone = TGAT(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features,
                                neighbor_sampler=train_neighbor_sampler,
                                time_feat_dim=args.time_feat_dim, num_layers=args.num_layers, num_heads=args.num_heads,
                                dropout=args.dropout, device=args.device)
    elif args.model_name in ['JODIE', 'DyRep', 'TGN']:
        # four floats that represent the mean and standard deviation of source and destination node time shifts in the training data, which is used for JODIE
        src_node_mean_time_shift, src_node_std_time_shift, dst_node_mean_time_shift_dst, dst_node_std_time_shift = \
            compute_src_dst_node_time_shifts(pretrain_data.src_node_ids, pretrain_data.dst_node_ids,
                                             pretrain_data.node_interact_times)
        dynamic_backbone = MemoryModel(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features,
                                       neighbor_sampler=train_neighbor_sampler,
                                       time_feat_dim=args.time_feat_dim, model_name=args.model_name,
                                       num_layers=args.num_layers, num_heads=args.num_heads,
                                       dropout=args.dropout, src_node_mean_time_shift=src_node_mean_time_shift,
                                       src_node_std_time_shift=src_node_std_time_shift,
                                       dst_node_mean_time_shift_dst=dst_node_mean_time_shift_dst,
                                       dst_node_std_time_shift=dst_node_std_time_shift, device=args.device)
    elif args.model_name == 'CAWN':
        dynamic_backbone = CAWN(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features,
                                neighbor_sampler=train_neighbor_sampler,
                                time_feat_dim=args.time_feat_dim, position_feat_dim=args.position_feat_dim,
                                walk_length=args.walk_length,
                                num_walk_heads=args.num_walk_heads, dropout=args.dropout, device=args.device)
    elif args.model_name == 'TCL':
        dynamic_backbone = TCL(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features,
                               neighbor_sampler=train_neighbor_sampler,
                               time_feat_dim=args.time_feat_dim, num_layers=args.num_layers, num_heads=args.num_heads,
                               num_depths=args.num_neighbors + 1, dropout=args.dropout, device=args.device)
    elif args.model_name == 'GraphMixer':
        dynamic_backbone = GraphMixer(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features,
                                      neighbor_sampler=train_neighbor_sampler,
                                      time_feat_dim=args.time_feat_dim, num_tokens=args.num_neighbors,
                                      num_layers=args.num_layers, dropout=args.dropout, device=args.device)
    elif args.model_name == 'DyGFormer':
        dynamic_backbone = DyGFormer(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features,
                                     neighbor_sampler=train_neighbor_sampler,
                                     time_feat_dim=args.time_feat_dim, channel_embedding_dim=args.channel_embedding_dim,
                                     patch_size=args.patch_size,
                                     num_layers=args.num_layers, num_heads=args.num_heads, dropout=args.dropout,
                                     max_input_sequence_length=args.max_input_sequence_length, device=args.device)
    elif args.model_name == 'DF2':
        dynamic_backbone = DF2(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features,
                               neighbor_sampler=train_neighbor_sampler,
                               time_feat_dim=args.time_feat_dim, channel_embedding_dim=args.channel_embedding_dim,
                               patch_size=args.patch_size,
                               num_layers=args.num_layers, num_heads=args.num_heads, dropout=args.dropout,
                               max_input_sequence_length=args.max_input_sequence_length, device=args.device)
    elif args.model_name == 'DF':
        dynamic_backbone = DF(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=train_neighbor_sampler,
                                    time_feat_dim=args.time_feat_dim, embedding_dim=args.embedding_dim, patch_size=args.patch_size,
                                   num_layers=args.num_layers,
                                    dropout=args.dropout, num_neighbors=args.num_neighbors, device=args.device)
    else:
        raise ValueError(f"Wrong value for model_name {args.model_name}!")


    link_predictor = MergeLayer(input_dim1=edge_raw_features.shape[1], input_dim2=edge_raw_features.shape[1],
                                hidden_dim=edge_raw_features.shape[1], output_dim=1)

    model = dynamic_backbone
    # model = nn.Sequential(dynamic_backbone, link_predictor)

    logger.info(f'model -> {model}')
    logger.info(f'model name: {args.model_name}, #parameters: {get_parameter_sizes(model) * 4} B, '
                f'{get_parameter_sizes(model) * 4 / 1024} KB, {get_parameter_sizes(model) * 4 / 1024 / 1024} MB.')

    optimizer = create_optimizer(model=model, optimizer_name=args.optimizer, learning_rate=args.learning_rate, weight_decay=args.weight_decay)

    model = convert_to_gpu(model, device=args.device)

    save_model_folder = f"./saved_models/{args.model_name}/{args.dataset_name}/{args.save_model_name}/"
    shutil.rmtree(save_model_folder, ignore_errors=True)
    os.makedirs(save_model_folder, exist_ok=True)

    save_model_path = os.path.join(save_model_folder, f"{args.save_model_name}.pkl")
    if args.model_name in ['JODIE', 'DyRep', 'TGN']:
        # path to additionally save the nonparametric data (e.g., tensors) in memory-based models (e.g., JODIE, DyRep, TGN)
        args.save_model_nonparametric_data_path = os.path.join(save_model_folder,
                                                               f"{args.save_model_name}_nonparametric_data.pkl")

    # early_stopping = EarlyStopping(patience=args.patience, save_model_folder=save_model_folder,
    #                                save_model_name=args.save_model_name, logger=logger, model_name=args.model_name)

    loss_func = nn.BCELoss()

    num_params_to_update = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters to update: {num_params_to_update}")


    train_loss_min = 1000000

    model.train()
    # pretrain
    for epoch in range(args.num_epochs):

        if args.model_name in ['DyRep', 'TGAT', 'TGN', 'CAWN', 'TCL', 'GraphMixer', 'DyGFormer', 'DF2', 'DF']:
            # training, only use training graph
            model.set_neighbor_sampler(train_neighbor_sampler)
            # model[0].set_neighbor_sampler(train_neighbor_sampler)

        if args.model_name in ['JODIE', 'DyRep', 'TGN']:
            # reinitialize memory of memory-based models at the start of each epoch
            model.memory_bank.__init_memory_bank__()

        # store train losses and metrics
        train_losses, train_metrics = [], []
        train_idx_data_loader_tqdm = tqdm(train_idx_data_loader, ncols=120)

        for batch_idx, train_data_indices in enumerate(train_idx_data_loader_tqdm):
            train_data_indices = train_data_indices.numpy()
            batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times, batch_edge_ids = \
                pretrain_data.src_node_ids[train_data_indices], pretrain_data.dst_node_ids[train_data_indices], \
                pretrain_data.node_interact_times[train_data_indices], pretrain_data.edge_ids[train_data_indices]

            _, batch_neg_dst_node_ids = train_neg_edge_sampler.sample(size=len(batch_src_node_ids))
            batch_neg_src_node_ids = batch_src_node_ids

            # we need to compute for positive and negative edges respectively, because the new sampling strategy (for evaluation) allows the negative source nodes to be
            # different from the source nodes, this is different from previous works that just replace destination nodes with negative destination nodes
            if args.model_name in ['TGAT', 'CAWN', 'TCL']:
                # get temporal embedding of source and destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch_src_node_embeddings, batch_dst_node_embeddings,_ = \
                    model.compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                      dst_node_ids=batch_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times,
                                                                      num_neighbors=args.num_neighbors)

                # get temporal embedding of negative source and negative destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings,_ = \
                    model.compute_src_dst_node_temporal_embeddings(src_node_ids=batch_neg_src_node_ids,
                                                                      dst_node_ids=batch_neg_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times,
                                                                      num_neighbors=args.num_neighbors)
            elif args.model_name in ['JODIE', 'DyRep', 'TGN']:
                # note that negative nodes do not change the memories while the positive nodes change the memories,
                # we need to first compute the embeddings of negative nodes for memory-based models
                # get temporal embedding of negative source and negative destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings,_ = \
                    model.compute_src_dst_node_temporal_embeddings(src_node_ids=batch_neg_src_node_ids,
                                                                      dst_node_ids=batch_neg_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times,
                                                                      edge_ids=None,
                                                                      edges_are_positive=False,
                                                                      num_neighbors=args.num_neighbors)

                # get temporal embedding of source and destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch_src_node_embeddings, batch_dst_node_embeddings,_ = \
                    model.compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                      dst_node_ids=batch_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times,
                                                                      edge_ids=batch_edge_ids,
                                                                      edges_are_positive=True,
                                                                      num_neighbors=args.num_neighbors)
            elif args.model_name in ['GraphMixer']:
                # get temporal embedding of source and destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch_src_node_embeddings, batch_dst_node_embeddings,_ = \
                    model.compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                      dst_node_ids=batch_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times,
                                                                      num_neighbors=args.num_neighbors,
                                                                      time_gap=args.time_gap)

                # get temporal embedding of negative source and negative destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings,_ = \
                    model.compute_src_dst_node_temporal_embeddings(src_node_ids=batch_neg_src_node_ids,
                                                                      dst_node_ids=batch_neg_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times,
                                                                      num_neighbors=args.num_neighbors,
                                                                      time_gap=args.time_gap)
            elif args.model_name in ['DyGFormer', 'DF2']:
                # get temporal embedding of source and destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch_src_node_embeddings, batch_dst_node_embeddings,_,_ = \
                    model.compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                      dst_node_ids=batch_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times)

                # get temporal embedding of negative source and negative destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings,_,_ = \
                    model.compute_src_dst_node_temporal_embeddings(src_node_ids=batch_neg_src_node_ids,
                                                                      dst_node_ids=batch_neg_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times)
            elif args.model_name in ['DF']:
                # (b, 2l, 4 * emb_dim)   input embedding to mamba
                # src_nodes_combine_fea, dst_nodes_combine_fea = model.get_src_des_node_combine_fea(batch_src_node_ids,
                #                                                                                  batch_dst_node_ids,
                #                                                                                  batch_node_interact_times)
                patches_data,neighbor_time,src_num_patches = model.get_src_des_node_combine_fea(batch_src_node_ids, batch_dst_node_ids,
                                                                     batch_node_interact_times)

                # get temporal embedding of source and destination nodes
                # Tensor, shape (batch_size, node_feat_dim=172)
                # batch_src_node_embeddings, batch_dst_node_embeddings = model.compute_src_dst_node_temporal_embeddings(
                #                                             src_nodes_combine_fea,dst_nodes_combine_fea)
                batch_src_node_embeddings, batch_dst_node_embeddings = model[
                    0].compute_src_dst_node_temporal_embeddings(patches_data, src_num_patches)

                # (b, 2l, 4 * emb_dim)   input embedding to mamba
                # neg_src_nodes_combine_fea, neg_dst_nodes_combine_fea = model.get_src_des_node_combine_fea(batch_neg_src_node_ids,
                #                                                                                   batch_neg_dst_node_ids,
                #                                                                                   batch_node_interact_times)
                patches_data,neighbor_time,src_num_patches = model.get_src_des_node_combine_fea(batch_neg_src_node_ids, batch_neg_dst_node_ids,
                                                                  batch_node_interact_times)
                # get temporal embedding of negative source and negative destination nodes
                # Tensor, shape (batch_size, node_feat_dim=172)
                batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = model[
                    0].compute_src_dst_node_temporal_embeddings(patches_data, src_num_patches)


            else:
                raise ValueError(f"Wrong value for model_name {args.model_name}!")
            # are_equal = torch.equal(patches_data1[:,:30,:], patches_data2[:,:30,:])
            # print(are_equal)
            # are_equal = torch.equal(batch_src_node_embeddings, batch_neg_src_node_embeddings)
            # print(are_equal)
            # pretrain loss
            if torch.isnan(batch_src_node_embeddings).any():
                print("src contains NaN values.")
            if torch.isnan(batch_dst_node_embeddings).any():
                print("des contains NaN values.")
            if torch.isnan(batch_neg_dst_node_embeddings).any():
                print("neg contains NaN values.")

            loss = contrastive_loss(batch_src_node_embeddings,batch_dst_node_embeddings,
                                    batch_neg_src_node_embeddings,batch_neg_dst_node_embeddings,tau=args.tau)
            # positive_probabilities = model[1](input_1=batch_src_node_embeddings,
            #                                   input_2=batch_dst_node_embeddings).squeeze(dim=-1).sigmoid()
            # negative_probabilities = model[1](input_1=batch_neg_src_node_embeddings,
            #                                   input_2=batch_neg_dst_node_embeddings).squeeze(dim=-1).sigmoid()
            #
            # predicts = torch.cat([positive_probabilities, negative_probabilities], dim=0)
            # labels = torch.cat([torch.ones_like(positive_probabilities), torch.zeros_like(negative_probabilities)],
            #                    dim=0)
            #
            # loss = loss_func(input=predicts, target=labels)
            #
            # train_metrics.append(get_link_prediction_metrics(predicts=predicts, labels=labels))



            train_losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            train_idx_data_loader_tqdm.set_description(f'Epoch: {epoch + 1}, train for the {batch_idx + 1}-th batch, train loss: {loss.item()}')

            if args.model_name in ['JODIE', 'DyRep', 'TGN']:
                # detach the memories and raw messages of nodes in the memory bank after each batch, so we don't back propagate to the start of time
                model.memory_bank.detach_memory_bank()

        # val_losses, val_metrics = evaluate_model_link_prediction(model_name=args.model_name,
        #                                                          model=model,
        #                                                          neighbor_sampler=full_neighbor_sampler,
        #                                                          evaluate_idx_data_loader=val_idx_data_loader,
        #                                                          evaluate_neg_edge_sampler=val_neg_edge_sampler,
        #                                                          evaluate_data=val_data,
        #                                                          loss_func=loss_func,
        #                                                          num_neighbors=args.num_neighbors,
        #                                                          time_gap=args.time_gap)
        #
        # new_node_val_losses, new_node_val_metrics = evaluate_model_link_prediction(model_name=args.model_name,
        #                                                                            model=model,
        #                                                                            neighbor_sampler=full_neighbor_sampler,
        #                                                                            evaluate_idx_data_loader=new_node_val_idx_data_loader,
        #                                                                            evaluate_neg_edge_sampler=new_node_val_neg_edge_sampler,
        #                                                                            evaluate_data=new_node_val_data,
        #                                                                            loss_func=loss_func,
        #                                                                            num_neighbors=args.num_neighbors,
        #                                                                            time_gap=args.time_gap)


        logger.info(f'Epoch: {epoch + 1}, learning rate: {optimizer.param_groups[0]["lr"]}, train loss: {np.mean(train_losses):.4f}')
        # for metric_name in train_metrics[0].keys():
        #     logger.info(f'train {metric_name}, {np.mean([train_metric[metric_name] for train_metric in train_metrics]):.4f}')
        # logger.info(f'validate loss: {np.mean(val_losses):.4f}')
        # for metric_name in val_metrics[0].keys():
        #     logger.info(f'validate {metric_name}, {np.mean([val_metric[metric_name] for val_metric in val_metrics]):.4f}')
        # logger.info(f'new node validate loss: {np.mean(new_node_val_losses):.4f}')
        # for metric_name in new_node_val_metrics[0].keys():
        #     logger.info(f'new node validate {metric_name}, {np.mean([new_node_val_metric[metric_name] for new_node_val_metric in new_node_val_metrics]):.4f}')

        # # select the best model based on all the validate metrics
        # val_metric_indicator = []
        # for metric_name in val_metrics[0].keys():
        #     val_metric_indicator.append(
        #         (metric_name, np.mean([val_metric[metric_name] for val_metric in val_metrics]), True))
        # early_stop = early_stopping.step(val_metric_indicator, model)
        #
        # if early_stop:
        #     break

        epoch_loss = np.mean(train_losses)
        if train_loss_min > epoch_loss:
            train_loss_min = epoch_loss
            torch.save(model.state_dict(),save_model_path)
            if args.model_name in ['JODIE', 'DyRep', 'TGN']:
                torch.save(model.memory_bank.node_raw_messages, args.save_model_nonparametric_data_path)
            logger.info(f"-----------save model {save_model_path} !--------------")

    sys.exit()
