import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import logging
import time
import argparse
import os
import json

from models.EdgeBank import edge_bank_link_prediction
from utils.metrics import get_link_prediction_metrics, get_node_classification_metrics
from utils.utils import set_random_seed
from utils.utils import NegativeEdgeSampler, NeighborSampler
from utils.DataLoader import Data

def l2_regularization(model, l2_lambda):
    return sum(torch.norm(param, 2) ** 2 for param in model.parameters()) * l2_lambda


def evaluate_model_link_prediction(model_name: str, model: nn.Module, neighbor_sampler: NeighborSampler, evaluate_idx_data_loader: DataLoader,
                                   evaluate_neg_edge_sampler: NegativeEdgeSampler, evaluate_data: Data, loss_func: nn.Module,
                                   num_neighbors: int = 20, time_gap: int = 2000,use_tig_prompt:bool = False,use_dyg_prompt=False,use_ddg_prompt=False):
    """
    evaluate models on the link prediction task
    :param model_name: str, name of the model
    :param model: nn.Module, the model to be evaluated
    :param neighbor_sampler: NeighborSampler, neighbor sampler
    :param evaluate_idx_data_loader: DataLoader, evaluate index data loader
    :param evaluate_neg_edge_sampler: NegativeEdgeSampler, evaluate negative edge sampler
    :param evaluate_data: Data, data to be evaluated
    :param loss_func: nn.Module, loss function
    :param num_neighbors: int, number of neighbors to sample for each node
    :param time_gap: int, time gap for neighbors to compute node features
    :return:
    """
    # Ensures the random sampler uses a fixed seed for evaluation (i.e. we always sample the same negatives for validation / test set)
    assert evaluate_neg_edge_sampler.seed is not None
    evaluate_neg_edge_sampler.reset_random_state()

    if model_name in ['DyRep', 'TGAT', 'TGN', 'CAWN', 'TCL', 'GraphMixer', 'DyGFormer','DF2','Dygamba']:
        # evaluation phase use all the graph information
        model[0].set_neighbor_sampler(neighbor_sampler)

    model.eval()

    with torch.no_grad():
        # store evaluate losses and metrics
        evaluate_losses, evaluate_metrics = [], []
        evaluate_idx_data_loader_tqdm = tqdm(evaluate_idx_data_loader, ncols=120)
        for batch_idx, evaluate_data_indices in enumerate(evaluate_idx_data_loader_tqdm):
            evaluate_data_indices = evaluate_data_indices.numpy()
            batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times, batch_edge_ids = \
                evaluate_data.src_node_ids[evaluate_data_indices],  evaluate_data.dst_node_ids[evaluate_data_indices], \
                evaluate_data.node_interact_times[evaluate_data_indices], evaluate_data.edge_ids[evaluate_data_indices]

            if evaluate_neg_edge_sampler.negative_sample_strategy != 'random':
                batch_neg_src_node_ids, batch_neg_dst_node_ids = evaluate_neg_edge_sampler.sample(size=len(batch_src_node_ids),
                                                                                                  batch_src_node_ids=batch_src_node_ids,
                                                                                                  batch_dst_node_ids=batch_dst_node_ids,
                                                                                                  current_batch_start_time=batch_node_interact_times[0],
                                                                                                  current_batch_end_time=batch_node_interact_times[-1])
            else:
                _, batch_neg_dst_node_ids = evaluate_neg_edge_sampler.sample(size=len(batch_src_node_ids))
                batch_neg_src_node_ids = batch_src_node_ids

            l2_loss = None

            # we need to compute for positive and negative edges respectively, because the new sampling strategy (for evaluation) allows the negative source nodes to be
            # different from the source nodes, this is different from previous works that just replace destination nodes with negative destination nodes
            if model_name in ['TGAT', 'CAWN', 'TCL']:
                if use_dyg_prompt:
                    src_padded_nodes_neighbor_node_raw_features, src_padded_nodes_edge_raw_features, src_padded_nodes_neighbor_time_features, src_nodes_neighbor_depth_features, \
                    dst_padded_nodes_neighbor_node_raw_features, dst_padded_nodes_edge_raw_features, dst_padded_nodes_neighbor_time_features, dst_nodes_neighbor_depth_features, neighbor_delta_times, \
                    src_neighbor_node_ids, dst_neighbor_node_ids = model[0].get_all_features(
                        src_node_ids=batch_src_node_ids, dst_node_ids=batch_dst_node_ids,
                        node_interact_times=batch_node_interact_times)

                    src_padded_nodes_neighbor_node_raw_features, src_padded_nodes_neighbor_time_features, dst_padded_nodes_neighbor_node_raw_features, dst_padded_nodes_neighbor_time_features = \
                    model[1].add_dyg_prompt_(src_padded_nodes_neighbor_node_raw_features,
                                             src_padded_nodes_neighbor_time_features,
                                             dst_padded_nodes_neighbor_node_raw_features,
                                             dst_padded_nodes_neighbor_time_features, )

                    batch_src_node_embeddings, batch_dst_node_embeddings = model[
                        0].compute_src_dst_node_temporal_embeddings_dyg(src_padded_nodes_neighbor_node_raw_features,
                                                                        src_padded_nodes_edge_raw_features,
                                                                        src_padded_nodes_neighbor_time_features,
                                                                        src_nodes_neighbor_depth_features, \
                                                                        dst_padded_nodes_neighbor_node_raw_features,
                                                                        dst_padded_nodes_edge_raw_features,
                                                                        dst_padded_nodes_neighbor_time_features,
                                                                        dst_nodes_neighbor_depth_features,
                                                                        neighbor_delta_times, src_neighbor_node_ids,
                                                                        dst_neighbor_node_ids)

                    neg_src_padded_nodes_neighbor_node_raw_features, neg_src_padded_nodes_edge_raw_features, neg_src_padded_nodes_neighbor_time_features, neg_src_nodes_neighbor_depth_features, \
                    neg_dst_padded_nodes_neighbor_node_raw_features, neg_dst_padded_nodes_edge_raw_features, neg_dst_padded_nodes_neighbor_time_features, neg_dst_nodes_neighbor_depth_features, neighbor_delta_times, src_neighbor_node_ids, dst_neighbor_node_ids = \
                        model[0].get_all_features(src_node_ids=batch_neg_src_node_ids,
                                                  dst_node_ids=batch_neg_dst_node_ids,
                                                  node_interact_times=batch_node_interact_times)

                    neg_src_padded_nodes_neighbor_node_raw_features, neg_src_padded_nodes_neighbor_time_features, neg_dst_padded_nodes_neighbor_node_raw_features, neg_dst_padded_nodes_neighbor_time_features = \
                        model[1].add_dyg_prompt_(neg_src_padded_nodes_neighbor_node_raw_features,
                                                 neg_src_padded_nodes_neighbor_time_features,
                                                 neg_dst_padded_nodes_neighbor_node_raw_features,
                                                 neg_dst_padded_nodes_neighbor_time_features)

                    batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = \
                        model[0].compute_src_dst_node_temporal_embeddings_dyg(
                            neg_src_padded_nodes_neighbor_node_raw_features, neg_src_padded_nodes_edge_raw_features,
                            neg_src_padded_nodes_neighbor_time_features,
                            neg_src_nodes_neighbor_depth_features, \
                            neg_dst_padded_nodes_neighbor_node_raw_features, neg_dst_padded_nodes_edge_raw_features,
                            neg_dst_padded_nodes_neighbor_time_features,
                            neg_dst_nodes_neighbor_depth_features,
                            neighbor_delta_times, src_neighbor_node_ids, dst_neighbor_node_ids)
                elif use_ddg_prompt:
                    src_data, dst_data, src_neighbor_node_ids, dst_neighbor_node_ids, \
                    src_neighbor_times, dst_neighbor_times = \
                        model[0].get_src_des_data(src_node_ids=batch_src_node_ids,
                                                  dst_node_ids=batch_dst_node_ids,
                                                  node_interact_times=batch_node_interact_times,
                                                  num_neighbors=num_neighbors)
                    src_data = model[1].add_mlp_weight_prompt(src_data, batch_node_interact_times, src_neighbor_times,
                                                              model[0].projection_layer.time)
                    dst_data = model[1].add_mlp_weight_prompt(dst_data, batch_node_interact_times,
                                                              dst_neighbor_times,
                                                              model[0].projection_layer.time)
                    batch_src_node_embeddings, batch_dst_node_embeddings = \
                        model[0].compute_embedding(src_data, dst_data, src_neighbor_node_ids, dst_neighbor_node_ids)

                    neg_src_data, neg_dst_data, neg_src_neighbor_node_ids, neg_dst_neighbor_node_ids, \
                    neg_src_neighbor_times, neg_dst_neighbor_times = \
                        model[0].get_src_des_data(src_node_ids=batch_neg_src_node_ids,
                                                  dst_node_ids=batch_neg_dst_node_ids,
                                                  node_interact_times=batch_node_interact_times,
                                                  num_neighbors=num_neighbors)
                    neg_src_data = model[1].add_mlp_weight_prompt(neg_src_data, batch_node_interact_times,
                                                                  neg_src_neighbor_times,
                                                                  model[0].projection_layer.time)
                    neg_dst_data = model[1].add_mlp_weight_prompt(neg_dst_data, batch_node_interact_times,
                                                                  neg_dst_neighbor_times,
                                                                  model[0].projection_layer.time)
                    batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = \
                        model[0].compute_embedding(neg_src_data, neg_dst_data, neg_src_neighbor_node_ids,
                                                   neg_dst_neighbor_node_ids)

                else:
                    # get temporal embedding of source and destination nodes
                    # two Tensors, with shape (batch_size, node_feat_dim)
                    batch_src_node_embeddings, batch_dst_node_embeddings, neighbor_delta_times = \
                        model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                          dst_node_ids=batch_dst_node_ids,
                                                                          node_interact_times=batch_node_interact_times,
                                                                          num_neighbors=num_neighbors)
                    if use_tig_prompt:
                        batch_src_node_embeddings, batch_dst_node_embeddings = model[1].add_tig_prompt(
                            batch_src_node_embeddings, batch_dst_node_embeddings, neighbor_delta_times, model="tcl")

                    # get temporal embedding of negative source and negative destination nodes
                    # two Tensors, with shape (batch_size, node_feat_dim)
                    batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings, neighbor_delta_times = \
                        model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_neg_src_node_ids,
                                                                          dst_node_ids=batch_neg_dst_node_ids,
                                                                          node_interact_times=batch_node_interact_times,
                                                                          num_neighbors=num_neighbors)
                    if use_tig_prompt:
                        batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = model[1].add_tig_prompt(
                            batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings, neighbor_delta_times,
                            model="tcl")

            elif model_name in ['JODIE', 'DyRep', 'TGN']:
                # note that negative nodes do not change the memories while the positive nodes change the memories,
                # we need to first compute the embeddings of negative nodes for memory-based models
                # get temporal embedding of negative source and negative destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings, neighbor_delta_times = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_neg_src_node_ids,
                                                                      dst_node_ids=batch_neg_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times,
                                                                      edge_ids=None,
                                                                      edges_are_positive=False,
                                                                      num_neighbors=num_neighbors)
                if use_tig_prompt:
                    batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = model[1].add_tig_prompt(batch_neg_src_node_embeddings,
                                                                                    batch_neg_dst_node_embeddings,
                                                                                    neighbor_delta_times)

                # get temporal embedding of source and destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch_src_node_embeddings, batch_dst_node_embeddings, neighbor_delta_times = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                      dst_node_ids=batch_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times,
                                                                      edge_ids=batch_edge_ids,
                                                                      edges_are_positive=True,
                                                                      num_neighbors=num_neighbors)
                if use_tig_prompt:
                    batch_src_node_embeddings, batch_dst_node_embeddings = model[1].add_tig_prompt(batch_src_node_embeddings,
                                                                                    batch_dst_node_embeddings,
                                                                                    neighbor_delta_times)

            elif model_name in ['GraphMixer']:
                if use_dyg_prompt:
                    src_nodes_edge_raw_features, src_nodes_neighbor_time_features, \
                    src_time_gap_neighbor_node_ids, src_nodes_time_gap_neighbor_node_raw_features, src_node_feature, \
                    dst_nodes_edge_raw_features, dst_nodes_neighbor_time_features, \
                    dst_time_gap_neighbor_node_ids, dst_nodes_time_gap_neighbor_node_raw_features, dst_node_feature, \
                        = model[0].compute_src_dst_node_temporal_embeddings_dyg(src_node_ids=batch_src_node_ids,
                                                                                dst_node_ids=batch_dst_node_ids,
                                                                                node_interact_times=batch_node_interact_times,
                                                                                num_neighbors=num_neighbors,
                                                                                time_gap=time_gap)
                    src_node_feature, src_nodes_neighbor_time_features, \
                    dst_node_feature, dst_nodes_neighbor_time_features = \
                        model[1].add_dyg_prompt_(src_node_feature,
                                                 src_nodes_neighbor_time_features,
                                                 dst_node_feature,
                                                 dst_nodes_neighbor_time_features,"gm")
                    batch_src_node_embeddings, batch_dst_node_embeddings = model[
                        0].compute_src_dst_node_temporal_embeddings_dyg_(
                        src_nodes_edge_raw_features, src_nodes_neighbor_time_features, \
                        src_time_gap_neighbor_node_ids, src_nodes_time_gap_neighbor_node_raw_features, src_node_feature,
                        dst_nodes_edge_raw_features, dst_nodes_neighbor_time_features, \
                        dst_time_gap_neighbor_node_ids, dst_nodes_time_gap_neighbor_node_raw_features, dst_node_feature,
                        batch_src_node_ids,
                        batch_dst_node_ids
                    )

                    neg_src_nodes_edge_raw_features, neg_src_nodes_neighbor_time_features, \
                    neg_src_time_gap_neighbor_node_ids, neg_src_nodes_time_gap_neighbor_node_raw_features, neg_src_node_feature, \
                    neg_dst_nodes_edge_raw_features, neg_dst_nodes_neighbor_time_features, \
                    neg_dst_time_gap_neighbor_node_ids, neg_dst_nodes_time_gap_neighbor_node_raw_features, neg_dst_node_feature, \
                        = model[0].compute_src_dst_node_temporal_embeddings_dyg(src_node_ids=batch_neg_src_node_ids,
                                                                                dst_node_ids=batch_neg_dst_node_ids,
                                                                                node_interact_times=batch_node_interact_times,
                                                                                num_neighbors=num_neighbors,
                                                                                time_gap=time_gap)
                    neg_src_node_feature, neg_src_nodes_neighbor_time_features, \
                    neg_dst_node_feature, neg_dst_nodes_neighbor_time_features = \
                        model[1].add_dyg_prompt_(neg_src_node_feature,
                                                 neg_src_nodes_neighbor_time_features,
                                                 neg_dst_node_feature,
                                                 neg_dst_nodes_neighbor_time_features,"gm")
                    batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = model[
                        0].compute_src_dst_node_temporal_embeddings_dyg_(
                        neg_src_nodes_edge_raw_features, neg_src_nodes_neighbor_time_features, \
                        neg_src_time_gap_neighbor_node_ids, neg_src_nodes_time_gap_neighbor_node_raw_features,
                        neg_src_node_feature,
                        neg_dst_nodes_edge_raw_features, neg_dst_nodes_neighbor_time_features, \
                        neg_dst_time_gap_neighbor_node_ids, neg_dst_nodes_time_gap_neighbor_node_raw_features,
                        neg_dst_node_feature,
                        batch_neg_src_node_ids,
                        batch_neg_dst_node_ids
                    )
                elif use_ddg_prompt:
                    src_data, dst_data, src_neighbor_times, dst_neighbor_times = \
                        model[0].get_datas(src_node_ids=batch_src_node_ids,
                                           dst_node_ids=batch_dst_node_ids,
                                           node_interact_times=batch_node_interact_times,
                                           num_neighbors=num_neighbors,
                                           time_gap=time_gap)
                    src_data = model[1].add_mlp_weight_prompt(src_data, batch_node_interact_times,
                                                              src_neighbor_times, None)
                    dst_data = model[1].add_mlp_weight_prompt(dst_data, batch_node_interact_times,
                                                              dst_neighbor_times, None)
                    batch_src_node_embeddings, batch_dst_node_embeddings = \
                        model[0].compute_embeddings(src_data, dst_data, batch_src_node_ids,
                                                    batch_dst_node_ids, batch_node_interact_times, time_gap)

                    neg_src_data, neg_dst_data, neg_src_neighbor_times, neg_dst_neighbor_times, = \
                        model[0].get_datas(src_node_ids=batch_neg_src_node_ids,
                                           dst_node_ids=batch_neg_dst_node_ids,
                                           node_interact_times=batch_node_interact_times,
                                           num_neighbors=num_neighbors,
                                           time_gap=time_gap)
                    neg_src_data = model[1].add_mlp_weight_prompt(neg_src_data, batch_node_interact_times,
                                                                  neg_src_neighbor_times,
                                                                  None)
                    neg_dst_data = model[1].add_mlp_weight_prompt(neg_dst_data, batch_node_interact_times,
                                                                  neg_dst_neighbor_times,
                                                                  None)
                    batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = \
                        model[0].compute_embeddings(neg_src_data, neg_dst_data, batch_neg_src_node_ids,
                                                    batch_neg_dst_node_ids, batch_node_interact_times, time_gap)
                else:
                    # get temporal embedding of source and destination nodes
                    # two Tensors, with shape (batch_size, node_feat_dim)
                    batch_src_node_embeddings, batch_dst_node_embeddings, neighbor_delta_times = \
                        model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                          dst_node_ids=batch_dst_node_ids,
                                                                          node_interact_times=batch_node_interact_times,
                                                                          num_neighbors=num_neighbors,
                                                                          time_gap=time_gap)
                    if use_tig_prompt:
                        batch_src_node_embeddings, batch_dst_node_embeddings = model[1].add_tig_prompt(
                            batch_src_node_embeddings, batch_dst_node_embeddings, neighbor_delta_times,model="gm")

                    # get temporal embedding of negative source and negative destination nodes
                    # two Tensors, with shape (batch_size, node_feat_dim)
                    batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings, neighbor_delta_times = \
                        model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_neg_src_node_ids,
                                                                          dst_node_ids=batch_neg_dst_node_ids,
                                                                          node_interact_times=batch_node_interact_times,
                                                                          num_neighbors=num_neighbors,
                                                                          time_gap=time_gap)
                    if use_tig_prompt:
                        batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = model[1].add_tig_prompt(
                            batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings, neighbor_delta_times,model="gm")

            elif model_name in ['DyGFormer']:
                if use_dyg_prompt:
                    src_padded_nodes_neighbor_node_raw_features, src_padded_nodes_edge_raw_features, src_padded_nodes_neighbor_time_features, src_padded_nodes_neighbor_co_occurrence_features, \
                    dst_padded_nodes_neighbor_node_raw_features, dst_padded_nodes_edge_raw_features, dst_padded_nodes_neighbor_time_features, dst_padded_nodes_neighbor_co_occurrence_features, neighbor_delta_times = \
                    model[0].get_all_features(src_node_ids=batch_src_node_ids, dst_node_ids=batch_dst_node_ids,
                                              node_interact_times=batch_node_interact_times)

                    src_padded_nodes_neighbor_node_raw_features, src_padded_nodes_neighbor_time_features, dst_padded_nodes_neighbor_node_raw_features, dst_padded_nodes_neighbor_time_features = \
                    model[1].add_dyg_prompt_(src_padded_nodes_neighbor_node_raw_features,
                                             src_padded_nodes_neighbor_time_features,
                                             dst_padded_nodes_neighbor_node_raw_features,
                                             dst_padded_nodes_neighbor_time_features)

                    batch_src_node_embeddings, batch_dst_node_embeddings, neighbor_delta_times, src_num_patches = model[
                        0].compute_src_dst_node_temporal_embeddings_dyg(src_padded_nodes_neighbor_node_raw_features,
                                                                        src_padded_nodes_edge_raw_features,
                                                                        src_padded_nodes_neighbor_time_features,
                                                                        src_padded_nodes_neighbor_co_occurrence_features, \
                                                                        dst_padded_nodes_neighbor_node_raw_features,
                                                                        dst_padded_nodes_edge_raw_features,
                                                                        dst_padded_nodes_neighbor_time_features,
                                                                        dst_padded_nodes_neighbor_co_occurrence_features,
                                                                        neighbor_delta_times)

                    neg_src_padded_nodes_neighbor_node_raw_features, neg_src_padded_nodes_edge_raw_features, neg_src_padded_nodes_neighbor_time_features, neg_src_padded_nodes_neighbor_co_occurrence_features, \
                    neg_dst_padded_nodes_neighbor_node_raw_features, neg_dst_padded_nodes_edge_raw_features, neg_dst_padded_nodes_neighbor_time_features, neg_dst_padded_nodes_neighbor_co_occurrence_features, neighbor_delta_times = \
                        model[0].get_all_features(src_node_ids=batch_neg_src_node_ids,
                                                  dst_node_ids=batch_neg_dst_node_ids,
                                                  node_interact_times=batch_node_interact_times)

                    neg_src_padded_nodes_neighbor_node_raw_features, neg_src_padded_nodes_neighbor_time_features, neg_dst_padded_nodes_neighbor_node_raw_features, neg_dst_padded_nodes_neighbor_time_features = \
                        model[1].add_dyg_prompt_(neg_src_padded_nodes_neighbor_node_raw_features,
                                                 neg_src_padded_nodes_neighbor_time_features,
                                                 neg_dst_padded_nodes_neighbor_node_raw_features,
                                                 neg_dst_padded_nodes_neighbor_time_features)

                    batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings, neighbor_delta_times, src_num_patches = \
                        model[0].compute_src_dst_node_temporal_embeddings_dyg(
                            neg_src_padded_nodes_neighbor_node_raw_features, neg_src_padded_nodes_edge_raw_features,
                            neg_src_padded_nodes_neighbor_time_features,
                            neg_src_padded_nodes_neighbor_co_occurrence_features, \
                            neg_dst_padded_nodes_neighbor_node_raw_features, neg_dst_padded_nodes_edge_raw_features,
                            neg_dst_padded_nodes_neighbor_time_features,
                            neg_dst_padded_nodes_neighbor_co_occurrence_features,
                            neighbor_delta_times)
                else:
                    # get temporal embedding of source and destination nodes
                    # two Tensors, with shape (batch_size, node_feat_dim)
                    batch_src_node_embeddings, batch_dst_node_embeddings, neighbor_delta_times, src_num_patches = \
                        model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                          dst_node_ids=batch_dst_node_ids,
                                                                          node_interact_times=batch_node_interact_times)

                    if use_tig_prompt:
                        batch_src_node_embeddings, batch_dst_node_embeddings = model[1].add_tig_prompt(
                            batch_src_node_embeddings, batch_dst_node_embeddings, neighbor_delta_times, src_num_patches)

                    # get temporal embedding of negative source and negative destination nodes
                    # two Tensors, with shape (batch_size, node_feat_dim)
                    batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings, neighbor_delta_times, src_num_patches = \
                        model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_neg_src_node_ids,
                                                                          dst_node_ids=batch_neg_dst_node_ids,
                                                                          node_interact_times=batch_node_interact_times)
                    if use_tig_prompt:
                        batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = model[1].add_tig_prompt(
                            batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings, neighbor_delta_times,
                            src_num_patches)

            elif model_name in ['DF2']:
                # get temporal embedding of source and destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch_src_node_embeddings, batch_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                      dst_node_ids=batch_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times)

                # get temporal embedding of negative source and negative destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_neg_src_node_ids,
                                                                      dst_node_ids=batch_neg_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times)
            elif model_name in ['Dygamba']:
                # (b, l, 4 * emb_dim)
                patches_data,neighbor_time,src_num_patches = model[0].get_src_des_node_combine_fea(batch_src_node_ids,batch_dst_node_ids,batch_node_interact_times)
                # adaptive neighbour num prompt
                # (b, l, 4 * emb_dim)   input embedding to mamba
                patches_data = model[1].add_mlp_weight_prompt(patches_data,batch_node_interact_times,neighbor_time,
                                                              model[0].projection_layer.time)
                # get temporal embedding of source and destination nodes
                # Tensor, shape (batch_size, node_feat_dim=172)
                batch_src_node_embeddings, batch_dst_node_embeddings = model[0].compute_src_dst_node_temporal_embeddings(patches_data,src_num_patches)
                # downstream task prompt
                # Tensor, shape (batch_size, node_feat_dim=172)
                # batch_src_node_embeddings, batch_dst_node_embeddings = model[1].add_feature_prompt(batch_src_node_embeddings, batch_dst_node_embeddings,
                #                                                                                    patches_data,patches_data_a, padded_nodes_neighbor_ids)


                # (b, l, 4 * emb_dim)
                patches_data,neighbor_time,src_num_patches = model[0].get_src_des_node_combine_fea(batch_neg_src_node_ids, batch_neg_dst_node_ids,batch_node_interact_times)
                # adaptive neighbour num prompt
                # (b, l, 4 * emb_dim)   input embedding to mamba
                patches_data = model[1].add_mlp_weight_prompt(patches_data,batch_node_interact_times,neighbor_time,
                                                              model[0].projection_layer.time)
                # get temporal embedding of negative source and negative destination nodes
                # Tensor, shape (batch_size, node_feat_dim=172)
                batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = model[0].compute_src_dst_node_temporal_embeddings(patches_data,src_num_patches)
                # downstream task prompt
                # Tensor, shape (batch_size, node_feat_dim=172)
                # batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = model[1].add_feature_prompt(batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings,
                #                                                                                            patches_data,patches_data_a, padded_nodes_neighbor_ids)
                try:
                    l2_loss = l2_regularization(model[1].mlp_weight_prompt, 0.01)
                except:
                    pass
            else:
                raise ValueError(f"Wrong value for model_name {model_name}!")

            # 在这里把使用prompt和没使用prompt的测试集的节点embedding保存下来, 用于可视化
            if use_ddg_prompt and batch_src_node_embeddings.shape[0] == 200:
                # 保存使用prompt的测试集的节点embedding
                torch.save(batch_src_node_embeddings, f'{model_name}_src_node_embeddings.pt')
                torch.save(batch_dst_node_embeddings, f'{model_name}_dst_node_embeddings.pt')
                torch.save(batch_neg_dst_node_embeddings, f'{model_name}_neg_dst_node_embeddings.pt')
                print("使用prompt的测试集的节点embedding保存成功")
            elif not use_ddg_prompt and batch_src_node_embeddings.shape[0] == 200:   
                # 保存没使用prompt的测试集的节点embedding
                torch.save(batch_src_node_embeddings, f'{model_name}_src_node_embeddings_without_prompt.pt')
                torch.save(batch_dst_node_embeddings, f'{model_name}_dst_node_embeddings_without_prompt.pt')
                torch.save(batch_neg_dst_node_embeddings, f'{model_name}_neg_dst_node_embeddings_without_prompt.pt')
                print("没使用prompt的测试集的节点embedding保存成功")

            # get positive and negative probabilities, shape (batch_size, )
            if model_name == "Dygamba" or use_tig_prompt or use_dyg_prompt or use_ddg_prompt:
                positive_probabilities = model[2](input_1=batch_src_node_embeddings,
                                                  input_2=batch_dst_node_embeddings).squeeze(dim=-1).sigmoid()
                negative_probabilities = model[2](input_1=batch_neg_src_node_embeddings,
                                                  input_2=batch_neg_dst_node_embeddings).squeeze(dim=-1).sigmoid()

            else:
                positive_probabilities = model[1](input_1=batch_src_node_embeddings, input_2=batch_dst_node_embeddings).squeeze(dim=-1).sigmoid()
                negative_probabilities = model[1](input_1=batch_neg_src_node_embeddings, input_2=batch_neg_dst_node_embeddings).squeeze(dim=-1).sigmoid()

            predicts = torch.cat([positive_probabilities, negative_probabilities], dim=0)
            labels = torch.cat([torch.ones_like(positive_probabilities), torch.zeros_like(negative_probabilities)], dim=0)

            loss = loss_func(input=predicts, target=labels)
            if l2_loss:
                loss += l2_loss

            evaluate_losses.append(loss.item())

            evaluate_metrics.append(get_link_prediction_metrics(predicts=predicts, labels=labels))

            evaluate_idx_data_loader_tqdm.set_description(f'evaluate for the {batch_idx + 1}-th batch, evaluate loss: {loss.item()}')

    return evaluate_losses, evaluate_metrics


def evaluate_model_node_classification(model_name: str, model: nn.Module, neighbor_sampler: NeighborSampler, evaluate_idx_data_loader: DataLoader,
                                       evaluate_data: Data, loss_func: nn.Module, num_neighbors: int = 20, time_gap: int = 2000,use_tig_prompt:bool = False,use_dyg_prompt:bool = False,use_ddg_prompt=False):
    """
    evaluate models on the node classification task
    :param model_name: str, name of the model
    :param model: nn.Module, the model to be evaluated
    :param neighbor_sampler: NeighborSampler, neighbor sampler
    :param evaluate_idx_data_loader: DataLoader, evaluate index data loader
    :param evaluate_data: Data, data to be evaluated
    :param loss_func: nn.Module, loss function
    :param num_neighbors: int, number of neighbors to sample for each node
    :param time_gap: int, time gap for neighbors to compute node features
    :return:
    """
    if model_name in ['DyRep', 'TGAT', 'TGN', 'CAWN', 'TCL', 'GraphMixer', 'DyGFormer', 'DF2', 'Dygamba']:
        # evaluation phase use all the graph information
        model[0].set_neighbor_sampler(neighbor_sampler)

    model.eval()

    with torch.no_grad():
        # store evaluate losses, trues and predicts
        evaluate_total_loss, evaluate_y_trues, evaluate_y_predicts = 0.0, [], []
        evaluate_idx_data_loader_tqdm = tqdm(evaluate_idx_data_loader, ncols=120)
        for batch_idx, evaluate_data_indices in enumerate(evaluate_idx_data_loader_tqdm):
            evaluate_data_indices = evaluate_data_indices.numpy()
            batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times, batch_edge_ids, batch_labels = \
                evaluate_data.src_node_ids[evaluate_data_indices],  evaluate_data.dst_node_ids[evaluate_data_indices], \
                evaluate_data.node_interact_times[evaluate_data_indices], evaluate_data.edge_ids[evaluate_data_indices], evaluate_data.labels[evaluate_data_indices]

            l2_loss = None

            if model_name in ['TGAT', 'CAWN', 'TCL']:
                if use_dyg_prompt:
                    src_padded_nodes_neighbor_node_raw_features, src_padded_nodes_edge_raw_features, src_padded_nodes_neighbor_time_features, src_nodes_neighbor_depth_features, \
                    dst_padded_nodes_neighbor_node_raw_features, dst_padded_nodes_edge_raw_features, dst_padded_nodes_neighbor_time_features, dst_nodes_neighbor_depth_features, neighbor_delta_times, \
                    src_neighbor_node_ids, dst_neighbor_node_ids = model[0].get_all_features(
                        src_node_ids=batch_src_node_ids, dst_node_ids=batch_dst_node_ids,
                        node_interact_times=batch_node_interact_times)

                    src_padded_nodes_neighbor_node_raw_features, src_padded_nodes_neighbor_time_features, dst_padded_nodes_neighbor_node_raw_features, dst_padded_nodes_neighbor_time_features = \
                    model[1].add_dyg_prompt_(src_padded_nodes_neighbor_node_raw_features,
                                             src_padded_nodes_neighbor_time_features,
                                             dst_padded_nodes_neighbor_node_raw_features,
                                             dst_padded_nodes_neighbor_time_features, )

                    batch_src_node_embeddings, batch_dst_node_embeddings = model[
                        0].compute_src_dst_node_temporal_embeddings_dyg(src_padded_nodes_neighbor_node_raw_features,
                                                                        src_padded_nodes_edge_raw_features,
                                                                        src_padded_nodes_neighbor_time_features,
                                                                        src_nodes_neighbor_depth_features, \
                                                                        dst_padded_nodes_neighbor_node_raw_features,
                                                                        dst_padded_nodes_edge_raw_features,
                                                                        dst_padded_nodes_neighbor_time_features,
                                                                        dst_nodes_neighbor_depth_features,
                                                                        neighbor_delta_times, src_neighbor_node_ids,
                                                                        dst_neighbor_node_ids)
                elif use_ddg_prompt:
                    src_data, dst_data, src_neighbor_node_ids, dst_neighbor_node_ids, \
                    src_neighbor_times, dst_neighbor_times = \
                        model[0].get_src_des_data(src_node_ids=batch_src_node_ids,
                                                  dst_node_ids=batch_dst_node_ids,
                                                  node_interact_times=batch_node_interact_times,
                                                  num_neighbors=num_neighbors)
                    src_data = model[1].add_mlp_weight_prompt(src_data, batch_node_interact_times, src_neighbor_times,
                                                              model[0].projection_layer.time)
                    dst_data = model[1].add_mlp_weight_prompt(dst_data, batch_node_interact_times,
                                                              dst_neighbor_times,
                                                              model[0].projection_layer.time)
                    batch_src_node_embeddings, batch_dst_node_embeddings = \
                        model[0].compute_embedding(src_data, dst_data, src_neighbor_node_ids, dst_neighbor_node_ids)
                else:
                    # get temporal embedding of source and destination nodes
                    # two Tensors, with shape (batch_size, node_feat_dim)
                    batch_src_node_embeddings, batch_dst_node_embeddings, neighbor_delta_times = \
                        model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                          dst_node_ids=batch_dst_node_ids,
                                                                          node_interact_times=batch_node_interact_times,
                                                                          num_neighbors=num_neighbors)
                    if use_tig_prompt:
                        batch_src_node_embeddings, batch_dst_node_embeddings = model[1].add_tig_prompt(
                            batch_src_node_embeddings, batch_dst_node_embeddings, neighbor_delta_times, model="tcl")
            elif model_name in ['JODIE', 'DyRep', 'TGN']:
                # get temporal embedding of source and destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch_src_node_embeddings, batch_dst_node_embeddings , neighbor_delta_times= \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                      dst_node_ids=batch_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times,
                                                                      edge_ids=batch_edge_ids,
                                                                      edges_are_positive=True,
                                                                      num_neighbors=num_neighbors)
                if use_tig_prompt:
                    batch_src_node_embeddings, batch_dst_node_embeddings = model[1].add_tig_prompt(batch_src_node_embeddings,
                                                                                    batch_dst_node_embeddings,
                                                                                    neighbor_delta_times)
            elif model_name in ['GraphMixer']:
                if use_dyg_prompt:
                    src_nodes_edge_raw_features, src_nodes_neighbor_time_features, \
                    src_time_gap_neighbor_node_ids, src_nodes_time_gap_neighbor_node_raw_features, src_node_feature, \
                    dst_nodes_edge_raw_features, dst_nodes_neighbor_time_features, \
                    dst_time_gap_neighbor_node_ids, dst_nodes_time_gap_neighbor_node_raw_features, dst_node_feature, \
                        = model[0].compute_src_dst_node_temporal_embeddings_dyg(src_node_ids=batch_src_node_ids,
                                                                                dst_node_ids=batch_dst_node_ids,
                                                                                node_interact_times=batch_node_interact_times,
                                                                                num_neighbors=num_neighbors,
                                                                                time_gap=time_gap)
                    src_node_feature, src_nodes_neighbor_time_features, \
                    dst_node_feature, dst_nodes_neighbor_time_features = \
                        model[1].add_dyg_prompt_(src_node_feature,
                                                 src_nodes_neighbor_time_features,
                                                 dst_node_feature,
                                                 dst_nodes_neighbor_time_features,"gm")
                    batch_src_node_embeddings, batch_dst_node_embeddings = model[
                        0].compute_src_dst_node_temporal_embeddings_dyg_(
                        src_nodes_edge_raw_features, src_nodes_neighbor_time_features, \
                        src_time_gap_neighbor_node_ids, src_nodes_time_gap_neighbor_node_raw_features, src_node_feature,
                        dst_nodes_edge_raw_features, dst_nodes_neighbor_time_features, \
                        dst_time_gap_neighbor_node_ids, dst_nodes_time_gap_neighbor_node_raw_features, dst_node_feature,
                        batch_src_node_ids,
                        batch_dst_node_ids
                    )
                elif use_ddg_prompt:
                    src_data, dst_data, src_neighbor_times, dst_neighbor_times = \
                        model[0].get_datas(src_node_ids=batch_src_node_ids,
                                           dst_node_ids=batch_dst_node_ids,
                                           node_interact_times=batch_node_interact_times,
                                           num_neighbors=num_neighbors,
                                           time_gap=time_gap)
                    src_data = model[1].add_mlp_weight_prompt(src_data, batch_node_interact_times,
                                                              src_neighbor_times, None)
                    dst_data = model[1].add_mlp_weight_prompt(dst_data, batch_node_interact_times,
                                                              dst_neighbor_times, None)
                    batch_src_node_embeddings, batch_dst_node_embeddings = \
                        model[0].compute_embeddings(src_data, dst_data, batch_src_node_ids,
                                                    batch_dst_node_ids, batch_node_interact_times, time_gap)
                else:
                    # get temporal embedding of source and destination nodes
                    # two Tensors, with shape (batch_size, node_feat_dim)
                    batch_src_node_embeddings, batch_dst_node_embeddings, neighbor_delta_times = \
                        model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                          dst_node_ids=batch_dst_node_ids,
                                                                          node_interact_times=batch_node_interact_times,
                                                                          num_neighbors=num_neighbors,
                                                                          time_gap=time_gap)
                    if use_tig_prompt:
                        batch_src_node_embeddings, batch_dst_node_embeddings = model[1].add_tig_prompt(
                            batch_src_node_embeddings, batch_dst_node_embeddings, neighbor_delta_times,model="gm")
            elif model_name in ['DyGFormer']:
                if use_dyg_prompt:
                    src_padded_nodes_neighbor_node_raw_features, src_padded_nodes_edge_raw_features, src_padded_nodes_neighbor_time_features, src_padded_nodes_neighbor_co_occurrence_features, \
                    dst_padded_nodes_neighbor_node_raw_features, dst_padded_nodes_edge_raw_features, dst_padded_nodes_neighbor_time_features, dst_padded_nodes_neighbor_co_occurrence_features, neighbor_delta_times = \
                    model[0].get_all_features(src_node_ids=batch_src_node_ids, dst_node_ids=batch_dst_node_ids,
                                              node_interact_times=batch_node_interact_times)

                    src_padded_nodes_neighbor_node_raw_features, src_padded_nodes_neighbor_time_features, dst_padded_nodes_neighbor_node_raw_features, dst_padded_nodes_neighbor_time_features = \
                    model[1].add_dyg_prompt_(src_padded_nodes_neighbor_node_raw_features,
                                             src_padded_nodes_neighbor_time_features,
                                             dst_padded_nodes_neighbor_node_raw_features,
                                             dst_padded_nodes_neighbor_time_features)

                    batch_src_node_embeddings, batch_dst_node_embeddings, neighbor_delta_times, src_num_patches = model[
                        0].compute_src_dst_node_temporal_embeddings_dyg(src_padded_nodes_neighbor_node_raw_features,
                                                                        src_padded_nodes_edge_raw_features,
                                                                        src_padded_nodes_neighbor_time_features,
                                                                        src_padded_nodes_neighbor_co_occurrence_features, \
                                                                        dst_padded_nodes_neighbor_node_raw_features,
                                                                        dst_padded_nodes_edge_raw_features,
                                                                        dst_padded_nodes_neighbor_time_features,
                                                                        dst_padded_nodes_neighbor_co_occurrence_features,
                                                                        neighbor_delta_times)
                else:
                    # get temporal embedding of source and destination nodes
                    # two Tensors, with shape (batch_size, node_feat_dim)
                    batch_src_node_embeddings, batch_dst_node_embeddings, neighbor_delta_times, src_num_patches = \
                        model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                          dst_node_ids=batch_dst_node_ids,
                                                                          node_interact_times=batch_node_interact_times)

                    if use_tig_prompt:
                        batch_src_node_embeddings, batch_dst_node_embeddings = model[1].add_tig_prompt(
                            batch_src_node_embeddings, batch_dst_node_embeddings, neighbor_delta_times, src_num_patches)

            elif model_name in ['DF2']:
                # get temporal embedding of source and destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch_src_node_embeddings, batch_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                      dst_node_ids=batch_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times)
            elif model_name in ['Dygamba']:
                # (b, l, 4 * emb_dim)
                patches_data, neighbor_time, src_num_patches = model[0].get_src_des_node_combine_fea(
                    batch_src_node_ids, batch_dst_node_ids,
                    batch_node_interact_times)                # adaptive neighbour num prompt
                # (b, l, 4 * emb_dim)   input embedding to mamba
                patches_data = model[1].add_mlp_weight_prompt(patches_data, batch_node_interact_times,
                                                              neighbor_time,
                                                              model[0].projection_layer.time)
                # get temporal embedding of source and destination nodes
                # Tensor, shape (batch_size, node_feat_dim=172)
                batch_src_node_embeddings, batch_dst_node_embeddings = model[
                    0].compute_src_dst_node_temporal_embeddings(patches_data, src_num_patches)
                # downstream task prompt
                # Tensor, shape (batch_size, node_feat_dim=172)
                # batch_src_node_embeddings, batch_dst_node_embeddings = model[1].add_feature_prompt(batch_src_node_embeddings, batch_dst_node_embeddings)
                try:
                    l2_loss = l2_regularization(model[1].mlp_weight_prompt, 0.01)
                except:
                    pass
            else:
                raise ValueError(f"Wrong value for model_name {model_name}!")
            # get predicted probabilities, shape (batch_size, )
            # if model_name == "Dygamba" or args.tig_prompt:

            if model_name == "Dygamba"or use_tig_prompt or use_dyg_prompt or use_ddg_prompt:
                predicts = model[2](x=batch_src_node_embeddings).squeeze(dim=-1).sigmoid()
            else:
                predicts = model[1](x=batch_src_node_embeddings).squeeze(dim=-1).sigmoid()
            labels = torch.from_numpy(batch_labels).float().to(predicts.device)

            loss = loss_func(input=predicts, target=labels)
            if l2_loss:
                loss += l2_loss

            evaluate_total_loss += loss.item()

            evaluate_y_trues.append(labels)
            evaluate_y_predicts.append(predicts)

            evaluate_idx_data_loader_tqdm.set_description(f'evaluate for the {batch_idx + 1}-th batch, evaluate loss: {loss.item()}')

        evaluate_total_loss /= (batch_idx + 1)
        evaluate_y_trues = torch.cat(evaluate_y_trues, dim=0)
        evaluate_y_predicts = torch.cat(evaluate_y_predicts, dim=0)

        evaluate_metrics = get_node_classification_metrics(predicts=evaluate_y_predicts, labels=evaluate_y_trues)

    return evaluate_total_loss, evaluate_metrics


def evaluate_edge_bank_link_prediction(args: argparse.Namespace, train_data: Data, val_data: Data, test_idx_data_loader: DataLoader,
                                       test_neg_edge_sampler: NegativeEdgeSampler, test_data: Data):
    """
    evaluate the EdgeBank model for link prediction
    :param args: argparse.Namespace, configuration
    :param train_data: Data, train data
    :param val_data: Data, validation data
    :param test_idx_data_loader: DataLoader, test index data loader
    :param test_neg_edge_sampler: NegativeEdgeSampler, test negative edge sampler
    :param test_data: Data, test data
    :return:
    """
    # generate the train_validation split of the data: needed for constructing the memory for EdgeBank
    train_val_data = Data(src_node_ids=np.concatenate([train_data.src_node_ids, val_data.src_node_ids]),
                          dst_node_ids=np.concatenate([train_data.dst_node_ids, val_data.dst_node_ids]),
                          node_interact_times=np.concatenate([train_data.node_interact_times, val_data.node_interact_times]),
                          edge_ids=np.concatenate([train_data.edge_ids, val_data.edge_ids]),
                          labels=np.concatenate([train_data.labels, val_data.labels]))

    test_metric_all_runs = []

    for run in range(args.num_runs):

        set_random_seed(seed=run)

        args.seed = run
        args.save_result_name = f'{args.negative_sample_strategy}_negative_sampling_{args.model_name}_seed{args.seed}'

        # set up logger
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        os.makedirs(f"./logs/{args.model_name}/{args.dataset_name}/{args.save_result_name}/", exist_ok=True)
        # create file handler that logs debug and higher level messages
        fh = logging.FileHandler(f"./logs/{args.model_name}/{args.dataset_name}/{args.save_result_name}/{str(time.time())}.log")
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
        logger.info(f"********** Run {run + 1} starts. **********")

        logger.info(f'configuration is {args}')

        loss_func = nn.BCELoss()

        # evaluate EdgeBank
        logger.info(f'get final performance on dataset {args.dataset_name}...')

        # Ensures the random sampler uses a fixed seed for evaluation (i.e. we always sample the same negatives for validation / test set)
        assert test_neg_edge_sampler.seed is not None
        test_neg_edge_sampler.reset_random_state()

        test_losses, test_metrics = [], []
        test_idx_data_loader_tqdm = tqdm(test_idx_data_loader, ncols=120)

        for batch_idx, test_data_indices in enumerate(test_idx_data_loader_tqdm):
            test_data_indices = test_data_indices.numpy()
            batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times = \
                test_data.src_node_ids[test_data_indices], test_data.dst_node_ids[test_data_indices], \
                test_data.node_interact_times[test_data_indices]

            if test_neg_edge_sampler.negative_sample_strategy != 'random':
                batch_neg_src_node_ids, batch_neg_dst_node_ids = test_neg_edge_sampler.sample(size=len(batch_src_node_ids),
                                                                                              batch_src_node_ids=batch_src_node_ids,
                                                                                              batch_dst_node_ids=batch_dst_node_ids,
                                                                                              current_batch_start_time=batch_node_interact_times[0],
                                                                                              current_batch_end_time=batch_node_interact_times[-1])
            else:
                _, batch_neg_dst_node_ids = test_neg_edge_sampler.sample(size=len(batch_src_node_ids))
                batch_neg_src_node_ids = batch_src_node_ids

            positive_edges = (batch_src_node_ids, batch_dst_node_ids)
            negative_edges = (batch_neg_src_node_ids, batch_neg_dst_node_ids)

            # incorporate the testing data before the current batch to history_data, which is similar to memory-based models
            history_data = Data(src_node_ids=np.concatenate([train_val_data.src_node_ids, test_data.src_node_ids[: test_data_indices[0]]]),
                                dst_node_ids=np.concatenate([train_val_data.dst_node_ids, test_data.dst_node_ids[: test_data_indices[0]]]),
                                node_interact_times=np.concatenate([train_val_data.node_interact_times, test_data.node_interact_times[: test_data_indices[0]]]),
                                edge_ids=np.concatenate([train_val_data.edge_ids, test_data.edge_ids[: test_data_indices[0]]]),
                                labels=np.concatenate([train_val_data.labels, test_data.labels[: test_data_indices[0]]]))

            # perform link prediction for EdgeBank
            positive_probabilities, negative_probabilities = edge_bank_link_prediction(history_data=history_data,
                                                                                       positive_edges=positive_edges,
                                                                                       negative_edges=negative_edges,
                                                                                       edge_bank_memory_mode=args.edge_bank_memory_mode,
                                                                                       time_window_mode=args.time_window_mode,
                                                                                       time_window_proportion=args.test_ratio)

            predicts = torch.from_numpy(np.concatenate([positive_probabilities, negative_probabilities])).float()
            labels = torch.cat([torch.ones(len(positive_probabilities)), torch.zeros(len(negative_probabilities))], dim=0)

            loss = loss_func(input=predicts, target=labels)

            test_losses.append(loss.item())

            test_metrics.append(get_link_prediction_metrics(predicts=predicts, labels=labels))

            test_idx_data_loader_tqdm.set_description(f'test for the {batch_idx + 1}-th batch, test loss: {loss.item()}')

        # store the evaluation metrics at the current run
        test_metric_dict = {}

        logger.info(f'test loss: {np.mean(test_losses):.4f}')
        for metric_name in test_metrics[0].keys():
            average_test_metric = np.mean([test_metric[metric_name] for test_metric in test_metrics])
            logger.info(f'test {metric_name}, {average_test_metric:.4f}')
            test_metric_dict[metric_name] = average_test_metric

        single_run_time = time.time() - run_start_time
        logger.info(f'Run {run + 1} cost {single_run_time:.2f} seconds.')

        test_metric_all_runs.append(test_metric_dict)

        # avoid the overlap of logs
        if run < args.num_runs - 1:
            logger.removeHandler(fh)
            logger.removeHandler(ch)

        # save model result
        result_json = {
            "test metrics": {metric_name: f'{test_metric_dict[metric_name]:.4f}'for metric_name in test_metric_dict}
        }
        result_json = json.dumps(result_json, indent=4)

        save_result_folder = f"./saved_results/{args.model_name}/{args.dataset_name}"
        os.makedirs(save_result_folder, exist_ok=True)
        save_result_path = os.path.join(save_result_folder, f"{args.save_result_name}.json")
        with open(save_result_path, 'w') as file:
            file.write(result_json)
        logger.info(f'save negative sampling results at {save_result_path}')

    # store the average metrics at the log of the last run
    logger.info(f'metrics over {args.num_runs} runs:')

    for metric_name in test_metric_all_runs[0].keys():
        logger.info(f'test {metric_name}, {[test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs]}')
        logger.info(f'average test {metric_name}, {np.mean([test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs]):.4f} '
                    f'± {np.std([test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs], ddof=1):.4f}')
