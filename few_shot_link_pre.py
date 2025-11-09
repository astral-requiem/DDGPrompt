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

from models.prompt import Prompt,TIG_Prompt,dyg_prompt
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
from utils.DataLoader import get_idx_data_loader, get_link_prediction_data, load_all_spilt_data, \
    generate_few_shot_dataset, split_train_data, get_few_shot_link_prediction_data
from utils.EarlyStopping import EarlyStopping
from utils.load_configs import get_link_prediction_args


def l2_regularization(model, l2_lambda):
    return sum(torch.norm(param, 2) ** 2 for param in model.parameters()) * l2_lambda

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

    args.train_shotnum = 30
    args.val_shotnum = 30
    args.few_shot_tasknum = 100
    args.label_num = 2
    args.seed = 0

    args.val_ratio = 0.01
    args.test_ratio = 0.18
    args.pretrain_ratio = 0.8
    args.finetune_ratio = 0.01  # 80,1,1,18

    args.tig_prompt = False
    args.use_dyg_prompt = False
    args.use_ddg_prompt = False

    all_train_cost_time = 0

    # get data for training, validation and testing
    node_raw_features, edge_raw_features, full_data, pretrain_data, finetune_data, val_data, test_data, new_node_val_data, new_node_test_data = \
        get_few_shot_link_prediction_data(dataset_name=args.dataset_name, pretrain_ratio=0.8,finetune_size=70)

    # # few interaction
    # full_data, pretrain_data, finetune_data, val_data, test_data, new_node_val_data, new_node_test_data = \
    #     process_datasets(full_data, pretrain_data, finetune_data, val_data, test_data, new_node_val_data,
    #                      new_node_test_data, 100)

    train_data = finetune_data

    val_metric_all_runs, new_node_val_metric_all_runs, test_metric_all_runs, new_node_test_metric_all_runs = [], [], [], []

    for run in range(args.num_runs):


        # initialize training neighbor sampler to retrieve temporal graph
        train_neighbor_sampler = get_neighbor_sampler(data=train_data,
                                                      sample_neighbor_strategy=args.sample_neighbor_strategy,
                                                      time_scaling_factor=args.time_scaling_factor, seed=0)

        # initialize validation and test neighbor sampler to retrieve temporal graph
        full_neighbor_sampler = get_neighbor_sampler(data=full_data,
                                                     sample_neighbor_strategy=args.sample_neighbor_strategy,
                                                     time_scaling_factor=args.time_scaling_factor, seed=1)

        # initialize negative samplers, set seeds for validation and testing so negatives are the same across different runs
        # in the inductive setting, negatives are sampled only amongst other new nodes
        # train negative edge sampler does not need to specify the seed, but evaluation samplers need to do so
        train_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=train_data.src_node_ids,
                                                     dst_node_ids=train_data.dst_node_ids)
        val_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=full_data.src_node_ids,
                                                   dst_node_ids=full_data.dst_node_ids, seed=0)
        new_node_val_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=new_node_val_data.src_node_ids,
                                                            dst_node_ids=new_node_val_data.dst_node_ids, seed=1)
        test_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=full_data.src_node_ids,
                                                    dst_node_ids=full_data.dst_node_ids, seed=2)
        new_node_test_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=new_node_test_data.src_node_ids,
                                                             dst_node_ids=new_node_test_data.dst_node_ids, seed=3)

        # get data loaders
        train_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(train_data.src_node_ids))),
                                                    batch_size=args.batch_size, shuffle=False)
        val_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(val_data.src_node_ids))),
                                                  batch_size=args.batch_size, shuffle=False)
        new_node_val_idx_data_loader = get_idx_data_loader(
            indices_list=list(range(len(new_node_val_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)
        test_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(test_data.src_node_ids))),
                                                   batch_size=args.batch_size, shuffle=False)
        new_node_test_idx_data_loader = get_idx_data_loader(
            indices_list=list(range(len(new_node_test_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)

        set_random_seed(seed=run)

        # aaai_year = 2048        # 2048 80%预训练,少量交互节点模型
        aaai_year = 2046        # 2046 80%预训练

        args.seed = run
        # args.load_model_name = f'{args.model_name}_seed{aaai_year}'
        args.load_model_name = f'DyGFormer_seed{aaai_year}'
        args.save_model_name = f'link_prediction_{args.model_name}_seed{args.seed}'

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
        logger.info(f"********** Run {run + 1} starts. **********")

        logger.info(f'configuration is {args}')

        # create model
        if args.model_name == 'TGAT':
            dynamic_backbone = TGAT(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=train_neighbor_sampler,
                                    time_feat_dim=args.time_feat_dim, num_layers=args.num_layers, num_heads=args.num_heads, dropout=args.dropout, device=args.device)
        elif args.model_name in ['JODIE', 'DyRep', 'TGN']:
            # four floats that represent the mean and standard deviation of source and destination node time shifts in the training data, which is used for JODIE
            src_node_mean_time_shift, src_node_std_time_shift, dst_node_mean_time_shift_dst, dst_node_std_time_shift = \
                compute_src_dst_node_time_shifts(train_data.src_node_ids, train_data.dst_node_ids, train_data.node_interact_times)
            dynamic_backbone = MemoryModel(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=train_neighbor_sampler,
                                           time_feat_dim=args.time_feat_dim, model_name=args.model_name, num_layers=args.num_layers, num_heads=args.num_heads,
                                           dropout=args.dropout, src_node_mean_time_shift=src_node_mean_time_shift, src_node_std_time_shift=src_node_std_time_shift,
                                           dst_node_mean_time_shift_dst=dst_node_mean_time_shift_dst, dst_node_std_time_shift=dst_node_std_time_shift, device=args.device,
                                           use_dyg_prompt=args.use_dyg_prompt)
        elif args.model_name == 'CAWN':
            dynamic_backbone = CAWN(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=train_neighbor_sampler,
                                    time_feat_dim=args.time_feat_dim, position_feat_dim=args.position_feat_dim, walk_length=args.walk_length,
                                    num_walk_heads=args.num_walk_heads, dropout=args.dropout, device=args.device)
        elif args.model_name == 'TCL':
            dynamic_backbone = TCL(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=train_neighbor_sampler,
                                   time_feat_dim=args.time_feat_dim, num_layers=args.num_layers, num_heads=args.num_heads,
                                   num_depths=args.num_neighbors + 1, dropout=args.dropout, device=args.device)
        elif args.model_name == 'GraphMixer':
            dynamic_backbone = GraphMixer(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=train_neighbor_sampler,
                                          time_feat_dim=args.time_feat_dim, num_tokens=args.num_neighbors, num_layers=args.num_layers, dropout=args.dropout, device=args.device)
        elif args.model_name == 'DyGFormer':
            dynamic_backbone = DyGFormer(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=train_neighbor_sampler,
                                         time_feat_dim=args.time_feat_dim, channel_embedding_dim=args.channel_embedding_dim, patch_size=args.patch_size,
                                         num_layers=args.num_layers, num_heads=args.num_heads, dropout=args.dropout,
                                         max_input_sequence_length=args.max_input_sequence_length, device=args.device)
        elif args.model_name == 'DF2':
            dynamic_backbone = DF2(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=train_neighbor_sampler,
                                         time_feat_dim=args.time_feat_dim, channel_embedding_dim=args.channel_embedding_dim, patch_size=args.patch_size,
                                         num_layers=args.num_layers, num_heads=args.num_heads, dropout=args.dropout,
                                         max_input_sequence_length=args.max_input_sequence_length, device=args.device)
        elif args.model_name == 'DF':
            dynamic_backbone = DF(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=train_neighbor_sampler,
                                        time_feat_dim=args.time_feat_dim, embedding_dim=args.embedding_dim, patch_size=args.patch_size,
                                       num_layers=args.num_layers,
                                        dropout=args.dropout, num_neighbors=args.num_neighbors, device=args.device)
        else:
            raise ValueError(f"Wrong value for model_name {args.model_name}!")

        # load saved pretrain model
        load_model_folder = f"./saved_models/DyGFormer/{args.dataset_name}/{args.load_model_name}/"
        # load_model_folder = f"./saved_models/{args.model_name}/{args.dataset_name}/{args.load_model_name}/"
        early_stopping = EarlyStopping(patience=0, save_model_folder=load_model_folder,
                                       save_model_name=args.load_model_name, logger=logger,
                                       model_name=args.model_name)

        link_predictor = MergeLayer(input_dim1=edge_raw_features.shape[1], input_dim2=edge_raw_features.shape[1],
                                    hidden_dim=edge_raw_features.shape[1], output_dim=1)
        # model = nn.Sequential(dynamic_backbone, link_predictor)
        model = dynamic_backbone
        early_stopping.load_checkpoint(model,map_location=f"{args.device}")

        if args.model_name == "DF" or args.use_ddg_prompt:
            prompt = Prompt(time_feat_dim = args.time_feat_dim,channel_embedding_dim=args.channel_embedding_dim,num_channels=4,hidden_dim=8)
            if args.model_name == "TCL":    # TCL
                prompt = Prompt(time_feat_dim = args.time_feat_dim,channel_embedding_dim=172,num_channels=4,hidden_dim=8)
            if args.model_name == "GraphMixer":
                prompt = Prompt(time_feat_dim = args.time_feat_dim,channel_embedding_dim=172,num_channels=4,hidden_dim=8,model_name="gm")
            prompt.set_device(args.device)
            model = nn.Sequential(model, prompt, link_predictor)

        else:
            if args.tig_prompt:
                tig_Prompt = TIG_Prompt(node_feat_dim=edge_raw_features.shape[1],time_feat_dim=args.time_feat_dim)
                tig_Prompt.set_device(args.device)
                model = nn.Sequential(model, tig_Prompt, link_predictor)
            elif args.use_dyg_prompt:
                dygprompt = dyg_prompt(node_feat_dim=edge_raw_features.shape[1],time_feat_dim=args.time_feat_dim,alpha=2)
                dygprompt.set_device(args.device)
                model = nn.Sequential(model, dygprompt, link_predictor)
            else:
                model = nn.Sequential(model, link_predictor)


        logger.info(f'model -> {model}')
        logger.info(f'model name: {args.model_name}, #parameters: {get_parameter_sizes(model) * 4} B, '
                    f'{get_parameter_sizes(model) * 4 / 1024} KB, {get_parameter_sizes(model) * 4 / 1024 / 1024} MB.')

        # follow previous work, we freeze the dynamic_backbone and only optimize the node_classifier and prompt
        for param in model[0].parameters():
            param.requires_grad = False


        # TGN-dygprompt
        if args.model_name == "TGN" and args.use_dyg_prompt:
            for param in model[0].embedding_module.dyg_prompt.parameters():
                param.requires_grad = True

        optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()),
                                     lr=args.learning_rate, weight_decay=args.weight_decay)

        model = convert_to_gpu(model, device=args.device)

        save_model_folder = f"./saved_models/{args.model_name}/{args.dataset_name}/{args.save_model_name}/"
        shutil.rmtree(save_model_folder, ignore_errors=True)
        os.makedirs(save_model_folder, exist_ok=True)

        early_stopping = EarlyStopping(patience=args.patience, save_model_folder=save_model_folder,
                                       save_model_name=args.save_model_name, logger=logger, model_name=args.model_name)

        loss_func = nn.BCELoss()

        model[0].eval()
        model[1].train()
        if args.model_name == "DF" or args.tig_prompt or args.use_dyg_prompt or args.use_ddg_prompt:
            model[2].train()
        # model[2].eval()

        num_params_to_update = sum(p.numel() for p in model[1].parameters() if p.requires_grad)
        print(f"Number of model 1 parameters to update: {num_params_to_update}")
        # num_params_to_update = sum(p.numel() for p in model[2].parameters() if p.requires_grad)
        # print(f"Number of model 2 parameters to update: {num_params_to_update}")



        for epoch in range(args.num_epochs):

            # model.train()
            if args.model_name in ['DyRep', 'TGAT', 'TGN', 'CAWN', 'TCL', 'GraphMixer', 'DyGFormer', 'DF2', 'DF']:
                # training, only use training graph
                model[0].set_neighbor_sampler(train_neighbor_sampler)
            if args.model_name in ['JODIE', 'DyRep', 'TGN']:
                # reinitialize memory of memory-based models at the start of each epoch
                model[0].memory_bank.__init_memory_bank__()

            # store train losses and metrics
            train_losses, train_metrics = [], []
            train_idx_data_loader_tqdm = tqdm(train_idx_data_loader, ncols=120)
            for batch_idx, train_data_indices in enumerate(train_idx_data_loader_tqdm):
                train_data_indices = train_data_indices.numpy()
                batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times, batch_edge_ids = \
                    train_data.src_node_ids[train_data_indices], train_data.dst_node_ids[train_data_indices], \
                    train_data.node_interact_times[train_data_indices], train_data.edge_ids[train_data_indices]

                _, batch_neg_dst_node_ids = train_neg_edge_sampler.sample(size=len(batch_src_node_ids))
                batch_neg_src_node_ids = batch_src_node_ids

                l2_loss = None

                # we need to compute for positive and negative edges respectively, because the new sampling strategy (for evaluation) allows the negative source nodes to be
                # different from the source nodes, this is different from previous works that just replace destination nodes with negative destination nodes
                if args.model_name in ['TGAT', 'CAWN', 'TCL']:
                    if args.use_dyg_prompt:
                        src_padded_nodes_neighbor_node_raw_features, src_padded_nodes_edge_raw_features, src_padded_nodes_neighbor_time_features, src_nodes_neighbor_depth_features, \
                        dst_padded_nodes_neighbor_node_raw_features, dst_padded_nodes_edge_raw_features, dst_padded_nodes_neighbor_time_features, dst_nodes_neighbor_depth_features, neighbor_delta_times,\
                        src_neighbor_node_ids,dst_neighbor_node_ids = model[0].get_all_features(src_node_ids=batch_src_node_ids,dst_node_ids=batch_dst_node_ids,node_interact_times=batch_node_interact_times)

                        src_padded_nodes_neighbor_node_raw_features,src_padded_nodes_neighbor_time_features,dst_padded_nodes_neighbor_node_raw_features,dst_padded_nodes_neighbor_time_features = model[1].add_dyg_prompt_(src_padded_nodes_neighbor_node_raw_features,src_padded_nodes_neighbor_time_features,
                                                                                                                                                                                                                           dst_padded_nodes_neighbor_node_raw_features,dst_padded_nodes_neighbor_time_features,)

                        batch_src_node_embeddings, batch_dst_node_embeddings = model[0].compute_src_dst_node_temporal_embeddings_dyg(src_padded_nodes_neighbor_node_raw_features, src_padded_nodes_edge_raw_features, src_padded_nodes_neighbor_time_features, src_nodes_neighbor_depth_features, \
                        dst_padded_nodes_neighbor_node_raw_features, dst_padded_nodes_edge_raw_features, dst_padded_nodes_neighbor_time_features, dst_nodes_neighbor_depth_features, neighbor_delta_times,src_neighbor_node_ids,dst_neighbor_node_ids)




                        neg_src_padded_nodes_neighbor_node_raw_features, neg_src_padded_nodes_edge_raw_features, neg_src_padded_nodes_neighbor_time_features, neg_src_nodes_neighbor_depth_features, \
                        neg_dst_padded_nodes_neighbor_node_raw_features, neg_dst_padded_nodes_edge_raw_features, neg_dst_padded_nodes_neighbor_time_features, neg_dst_nodes_neighbor_depth_features, neighbor_delta_times,src_neighbor_node_ids,dst_neighbor_node_ids = \
                        model[0].get_all_features(src_node_ids=batch_neg_src_node_ids, dst_node_ids=batch_neg_dst_node_ids,
                                                  node_interact_times=batch_node_interact_times)

                        neg_src_padded_nodes_neighbor_node_raw_features,neg_src_padded_nodes_neighbor_time_features,neg_dst_padded_nodes_neighbor_node_raw_features,neg_dst_padded_nodes_neighbor_time_features = \
                        model[1].add_dyg_prompt_(neg_src_padded_nodes_neighbor_node_raw_features,neg_src_padded_nodes_neighbor_time_features,neg_dst_padded_nodes_neighbor_node_raw_features,neg_dst_padded_nodes_neighbor_time_features)

                        batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = \
                        model[0].compute_src_dst_node_temporal_embeddings_dyg(
                            neg_src_padded_nodes_neighbor_node_raw_features, neg_src_padded_nodes_edge_raw_features,
                            neg_src_padded_nodes_neighbor_time_features,neg_src_nodes_neighbor_depth_features, \
                            neg_dst_padded_nodes_neighbor_node_raw_features, neg_dst_padded_nodes_edge_raw_features,
                            neg_dst_padded_nodes_neighbor_time_features,neg_dst_nodes_neighbor_depth_features,
                            neighbor_delta_times,src_neighbor_node_ids,dst_neighbor_node_ids)
                    elif args.use_ddg_prompt:
                        src_data,dst_data,src_neighbor_node_ids,dst_neighbor_node_ids,\
                        src_neighbor_times,dst_neighbor_times = \
                            model[0].get_src_des_data(src_node_ids=batch_src_node_ids,
                                                                              dst_node_ids=batch_dst_node_ids,
                                                                              node_interact_times=batch_node_interact_times,
                                                                              num_neighbors=args.num_neighbors)
                        src_data = model[1].add_mlp_weight_prompt(src_data,batch_node_interact_times,src_neighbor_times,
                                                                  model[0].projection_layer.time)
                        dst_data = model[1].add_mlp_weight_prompt(dst_data, batch_node_interact_times,
                                                                  dst_neighbor_times,
                                                                  model[0].projection_layer.time)
                        batch_src_node_embeddings, batch_dst_node_embeddings = \
                            model[0].compute_embedding(src_data,dst_data,src_neighbor_node_ids,dst_neighbor_node_ids)



                        neg_src_data, neg_dst_data, neg_src_neighbor_node_ids, neg_dst_neighbor_node_ids, \
                        neg_src_neighbor_times, neg_dst_neighbor_times = \
                            model[0].get_src_des_data(src_node_ids=batch_neg_src_node_ids,
                                                      dst_node_ids=batch_neg_dst_node_ids,
                                                      node_interact_times=batch_node_interact_times,
                                                      num_neighbors=args.num_neighbors)
                        neg_src_data = model[1].add_mlp_weight_prompt(neg_src_data, batch_node_interact_times,
                                                                  neg_src_neighbor_times,
                                                                  model[0].projection_layer.time)
                        neg_dst_data = model[1].add_mlp_weight_prompt(neg_dst_data, batch_node_interact_times,
                                                                  neg_dst_neighbor_times,
                                                                  model[0].projection_layer.time)
                        batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = \
                            model[0].compute_embedding(neg_src_data, neg_dst_data, neg_src_neighbor_node_ids, neg_dst_neighbor_node_ids)

                    else:
                        # get temporal embedding of source and destination nodes
                        # two Tensors, with shape (batch_size, node_feat_dim)
                        batch_src_node_embeddings, batch_dst_node_embeddings, neighbor_delta_times = \
                            model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                              dst_node_ids=batch_dst_node_ids,
                                                                              node_interact_times=batch_node_interact_times,
                                                                              num_neighbors=args.num_neighbors)
                        if args.tig_prompt:
                            batch_src_node_embeddings, batch_dst_node_embeddings = model[1].add_tig_prompt(batch_src_node_embeddings, batch_dst_node_embeddings, neighbor_delta_times,model="tcl")


                        # get temporal embedding of negative source and negative destination nodes
                        # two Tensors, with shape (batch_size, node_feat_dim)
                        batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings, neighbor_delta_times = \
                            model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_neg_src_node_ids,
                                                                              dst_node_ids=batch_neg_dst_node_ids,
                                                                              node_interact_times=batch_node_interact_times,
                                                                              num_neighbors=args.num_neighbors)
                        if args.tig_prompt:
                            batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = model[1].add_tig_prompt(
                                batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings, neighbor_delta_times,
                                model="tcl")

                elif args.model_name in ['JODIE', 'DyRep', 'TGN']:
                    # note that negative nodes do not change the memories while the positive nodes change the memories,
                    # we need to first compute the embeddings of negative nodes for memory-based models
                    # get temporal embedding of source and destination nodes
                    # two Tensors, with shape (batch_size, node_feat_dim)
                    batch_src_node_embeddings, batch_dst_node_embeddings, neighbor_delta_times = \
                        model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                          dst_node_ids=batch_dst_node_ids,
                                                                          node_interact_times=batch_node_interact_times,
                                                                          edge_ids=batch_edge_ids,
                                                                          edges_are_positive=True,
                                                                          num_neighbors=args.num_neighbors)

                    if args.tig_prompt:
                        batch_src_node_embeddings, batch_dst_node_embeddings = model[1].add_tig_prompt(batch_src_node_embeddings, batch_dst_node_embeddings, neighbor_delta_times)

                    # get temporal embedding of negative source and negative destination nodes
                    # two Tensors, with shape (batch_size, node_feat_dim)
                    batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings, neighbor_delta_times = \
                        model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_neg_src_node_ids,
                                                                          dst_node_ids=batch_neg_dst_node_ids,
                                                                          node_interact_times=batch_node_interact_times,
                                                                          edge_ids=None,
                                                                          edges_are_positive=False,
                                                                          num_neighbors=args.num_neighbors)

                    if args.tig_prompt:
                        batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = model[1].add_tig_prompt(batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings, neighbor_delta_times)

                elif args.model_name in ['GraphMixer']:
                    if args.use_dyg_prompt:
                        src_nodes_edge_raw_features, src_nodes_neighbor_time_features, \
                        src_time_gap_neighbor_node_ids, src_nodes_time_gap_neighbor_node_raw_features,src_node_feature, \
                        dst_nodes_edge_raw_features, dst_nodes_neighbor_time_features, \
                        dst_time_gap_neighbor_node_ids, dst_nodes_time_gap_neighbor_node_raw_features, dst_node_feature,\
                            = model[0].compute_src_dst_node_temporal_embeddings_dyg(src_node_ids=batch_src_node_ids,
                                                                          dst_node_ids=batch_dst_node_ids,
                                                                          node_interact_times=batch_node_interact_times,
                                                                          num_neighbors=args.num_neighbors,
                                                                          time_gap=args.time_gap)
                        src_node_feature, src_nodes_neighbor_time_features,\
                        dst_node_feature, dst_nodes_neighbor_time_features = \
                        model[1].add_dyg_prompt_(src_node_feature,
                                                 src_nodes_neighbor_time_features,
                                                 dst_node_feature,
                                                 dst_nodes_neighbor_time_features,"gm")
                        batch_src_node_embeddings, batch_dst_node_embeddings = model[0].compute_src_dst_node_temporal_embeddings_dyg_(
                            src_nodes_edge_raw_features, src_nodes_neighbor_time_features, \
                            src_time_gap_neighbor_node_ids, src_nodes_time_gap_neighbor_node_raw_features,src_node_feature,
                            dst_nodes_edge_raw_features, dst_nodes_neighbor_time_features, \
                            dst_time_gap_neighbor_node_ids, dst_nodes_time_gap_neighbor_node_raw_features,dst_node_feature, batch_src_node_ids,
                            batch_dst_node_ids
                        )

                        neg_src_nodes_edge_raw_features, neg_src_nodes_neighbor_time_features, \
                        neg_src_time_gap_neighbor_node_ids, neg_src_nodes_time_gap_neighbor_node_raw_features,neg_src_node_feature, \
                        neg_dst_nodes_edge_raw_features, neg_dst_nodes_neighbor_time_features, \
                        neg_dst_time_gap_neighbor_node_ids, neg_dst_nodes_time_gap_neighbor_node_raw_features,neg_dst_node_feature,\
                            = model[0].compute_src_dst_node_temporal_embeddings_dyg(src_node_ids=batch_neg_src_node_ids,
                                                                                    dst_node_ids=batch_neg_dst_node_ids,
                                                                                    node_interact_times=batch_node_interact_times,
                                                                                    num_neighbors=args.num_neighbors,
                                                                                    time_gap=args.time_gap)
                        neg_src_node_feature, neg_src_nodes_neighbor_time_features, \
                        neg_dst_node_feature, neg_dst_nodes_neighbor_time_features = \
                            model[1].add_dyg_prompt_(neg_src_node_feature,
                                                     neg_src_nodes_neighbor_time_features,
                                                     neg_dst_node_feature,
                                                     neg_dst_nodes_neighbor_time_features,"gm")
                        batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = model[
                            0].compute_src_dst_node_temporal_embeddings_dyg_(
                            neg_src_nodes_edge_raw_features, neg_src_nodes_neighbor_time_features, \
                            neg_src_time_gap_neighbor_node_ids, neg_src_nodes_time_gap_neighbor_node_raw_features,neg_src_node_feature,
                            neg_dst_nodes_edge_raw_features, neg_dst_nodes_neighbor_time_features, \
                            neg_dst_time_gap_neighbor_node_ids, neg_dst_nodes_time_gap_neighbor_node_raw_features,neg_dst_node_feature,
                            batch_neg_src_node_ids,
                            batch_neg_dst_node_ids
                        )
                    elif args.use_ddg_prompt:
                        src_data, dst_data,src_neighbor_times,dst_neighbor_times = \
                            model[0].get_datas(src_node_ids=batch_src_node_ids,
                                                      dst_node_ids=batch_dst_node_ids,
                                                      node_interact_times=batch_node_interact_times,
                                                      num_neighbors=args.num_neighbors,
                                               time_gap=args.time_gap)
                        src_data = model[1].add_mlp_weight_prompt(src_data, batch_node_interact_times,
                                                                  src_neighbor_times,None)
                        dst_data = model[1].add_mlp_weight_prompt(dst_data, batch_node_interact_times,
                                                                  dst_neighbor_times,None)
                        batch_src_node_embeddings, batch_dst_node_embeddings = \
                            model[0].compute_embeddings(src_data, dst_data,batch_src_node_ids,
                                                        batch_dst_node_ids,batch_node_interact_times,args.time_gap)


                        neg_src_data, neg_dst_data,neg_src_neighbor_times,neg_dst_neighbor_times, = \
                            model[0].get_datas(src_node_ids=batch_neg_src_node_ids,
                                               dst_node_ids=batch_neg_dst_node_ids,
                                               node_interact_times=batch_node_interact_times,
                                               num_neighbors=args.num_neighbors,
                                               time_gap=args.time_gap)
                        neg_src_data = model[1].add_mlp_weight_prompt(neg_src_data, batch_node_interact_times,
                                                                      neg_src_neighbor_times,
                                                                      None)
                        neg_dst_data = model[1].add_mlp_weight_prompt(neg_dst_data, batch_node_interact_times,
                                                                      neg_dst_neighbor_times,
                                                                      None)
                        batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = \
                            model[0].compute_embeddings(neg_src_data, neg_dst_data, batch_neg_src_node_ids,
                                                       batch_neg_dst_node_ids,batch_node_interact_times,args.time_gap)



                    else:
                        # get temporal embedding of source and destination nodes
                        # two Tensors, with shape (batch_size, node_feat_dim)
                        batch_src_node_embeddings, batch_dst_node_embeddings ,neighbor_delta_times= \
                            model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                              dst_node_ids=batch_dst_node_ids,
                                                                              node_interact_times=batch_node_interact_times,
                                                                              num_neighbors=args.num_neighbors,
                                                                              time_gap=args.time_gap)
                        if args.tig_prompt:
                            batch_src_node_embeddings, batch_dst_node_embeddings = model[1].add_tig_prompt(batch_src_node_embeddings, batch_dst_node_embeddings, neighbor_delta_times,model="gm")


                        # get temporal embedding of negative source and negative destination nodes
                        # two Tensors, with shape (batch_size, node_feat_dim)
                        batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings,neighbor_delta_times = \
                            model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_neg_src_node_ids,
                                                                              dst_node_ids=batch_neg_dst_node_ids,
                                                                              node_interact_times=batch_node_interact_times,
                                                                              num_neighbors=args.num_neighbors,
                                                                              time_gap=args.time_gap)
                        if args.tig_prompt:
                            batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = model[1].add_tig_prompt(batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings, neighbor_delta_times,model="gm")

                elif args.model_name in ['DyGFormer','DF2']:
                    if args.use_dyg_prompt:
                        src_padded_nodes_neighbor_node_raw_features, src_padded_nodes_edge_raw_features, src_padded_nodes_neighbor_time_features, src_padded_nodes_neighbor_co_occurrence_features, \
                        dst_padded_nodes_neighbor_node_raw_features, dst_padded_nodes_edge_raw_features, dst_padded_nodes_neighbor_time_features, dst_padded_nodes_neighbor_co_occurrence_features, neighbor_delta_times = model[0].get_all_features(src_node_ids=batch_src_node_ids,dst_node_ids=batch_dst_node_ids,node_interact_times=batch_node_interact_times)

                        src_padded_nodes_neighbor_node_raw_features,src_padded_nodes_neighbor_time_features,dst_padded_nodes_neighbor_node_raw_features,dst_padded_nodes_neighbor_time_features = model[1].add_dyg_prompt_(src_padded_nodes_neighbor_node_raw_features,src_padded_nodes_neighbor_time_features,
                                                                                                                                                                                                                           dst_padded_nodes_neighbor_node_raw_features,dst_padded_nodes_neighbor_time_features)

                        batch_src_node_embeddings, batch_dst_node_embeddings, neighbor_delta_times, src_num_patches = model[0].compute_src_dst_node_temporal_embeddings_dyg(src_padded_nodes_neighbor_node_raw_features, src_padded_nodes_edge_raw_features, src_padded_nodes_neighbor_time_features, src_padded_nodes_neighbor_co_occurrence_features, \
                        dst_padded_nodes_neighbor_node_raw_features, dst_padded_nodes_edge_raw_features, dst_padded_nodes_neighbor_time_features, dst_padded_nodes_neighbor_co_occurrence_features, neighbor_delta_times)




                        neg_src_padded_nodes_neighbor_node_raw_features, neg_src_padded_nodes_edge_raw_features, neg_src_padded_nodes_neighbor_time_features, neg_src_padded_nodes_neighbor_co_occurrence_features, \
                        neg_dst_padded_nodes_neighbor_node_raw_features, neg_dst_padded_nodes_edge_raw_features, neg_dst_padded_nodes_neighbor_time_features, neg_dst_padded_nodes_neighbor_co_occurrence_features, neighbor_delta_times = \
                        model[0].get_all_features(src_node_ids=batch_neg_src_node_ids, dst_node_ids=batch_neg_dst_node_ids,
                                                  node_interact_times=batch_node_interact_times)

                        neg_src_padded_nodes_neighbor_node_raw_features,neg_src_padded_nodes_neighbor_time_features,neg_dst_padded_nodes_neighbor_node_raw_features,neg_dst_padded_nodes_neighbor_time_features = \
                        model[1].add_dyg_prompt_(neg_src_padded_nodes_neighbor_node_raw_features,neg_src_padded_nodes_neighbor_time_features,neg_dst_padded_nodes_neighbor_node_raw_features,neg_dst_padded_nodes_neighbor_time_features)

                        batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings, neighbor_delta_times, src_num_patches = \
                        model[0].compute_src_dst_node_temporal_embeddings_dyg(
                            neg_src_padded_nodes_neighbor_node_raw_features, neg_src_padded_nodes_edge_raw_features,
                            neg_src_padded_nodes_neighbor_time_features,neg_src_padded_nodes_neighbor_co_occurrence_features, \
                            neg_dst_padded_nodes_neighbor_node_raw_features, neg_dst_padded_nodes_edge_raw_features,
                            neg_dst_padded_nodes_neighbor_time_features,neg_dst_padded_nodes_neighbor_co_occurrence_features,
                            neighbor_delta_times)

                    else:
                        # get temporal embedding of source and destination nodes
                        # two Tensors, with shape (batch_size, node_feat_dim)
                        batch_src_node_embeddings, batch_dst_node_embeddings,neighbor_delta_times,src_num_patches = \
                            model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                              dst_node_ids=batch_dst_node_ids,
                                                                              node_interact_times=batch_node_interact_times)

                        if args.tig_prompt:
                            batch_src_node_embeddings, batch_dst_node_embeddings = model[1].add_tig_prompt(batch_src_node_embeddings, batch_dst_node_embeddings, neighbor_delta_times,src_num_patches)


                        # get temporal embedding of negative source and negative destination nodes
                        # two Tensors, with shape (batch_size, node_feat_dim)
                        batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings,neighbor_delta_times,src_num_patches = \
                            model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_neg_src_node_ids,
                                                                              dst_node_ids=batch_neg_dst_node_ids,
                                                                              node_interact_times=batch_node_interact_times)
                        if args.tig_prompt:
                            batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = model[1].add_tig_prompt(batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings, neighbor_delta_times,src_num_patches)


                elif args.model_name in ['DF']:
                    # (b, l, 4 * emb_dim)
                    patches_data,neighbor_time,src_num_patches = model[0].get_src_des_node_combine_fea(batch_src_node_ids, batch_dst_node_ids,
                                                                         batch_node_interact_times)
                    # adaptive neighbour num prompt
                    # (b, l, 4 * emb_dim)   input embedding to mamba
                    patches_data = model[1].add_mlp_weight_prompt(patches_data,batch_node_interact_times,neighbor_time,
                                                                  model[0].projection_layer.time)
                    # get temporal embedding of source and destination nodes
                    # Tensor, shape (batch_size, node_feat_dim=172)
                    batch_src_node_embeddings, batch_dst_node_embeddings = model[0].compute_src_dst_node_temporal_embeddings(patches_data,src_num_patches)
                    # downstream task prompt
                    # Tensor, shape (batch_size, node_feat_dim=172)
                    # batch_src_node_embeddings, batch_dst_node_embeddings = model[1].add_feature_prompt(
                    #     batch_src_node_embeddings, batch_dst_node_embeddings,patches_data,patches_data_a,padded_nodes_neighbor_ids)

                    # (b, l, 4 * emb_dim)
                    patches_data,neighbor_time,src_num_patches = model[0].get_src_des_node_combine_fea(batch_neg_src_node_ids,
                                                                         batch_neg_dst_node_ids,
                                                                         batch_node_interact_times)
                    # adaptive neighbour num prompt
                    # (b, l, 4 * emb_dim)   input embedding to mamba
                    patches_data = model[1].add_mlp_weight_prompt(patches_data,batch_node_interact_times,neighbor_time,
                                                                  model[0].projection_layer.time)
                    # get temporal embedding of negastive source and negative destination nodes
                    # Tensor, shape (batch_size, node_feat_dim=172)
                    batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = model[0].compute_src_dst_node_temporal_embeddings(patches_data,src_num_patches)
                    # downstream task prompt
                    # Tensor, shape (batch_size, node_feat_dim=172)
                    # batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = model[1].add_feature_prompt(
                    #     batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings, patches_data,patches_data_a,padded_nodes_neighbor_ids)
                    try:
                        l2_loss = l2_regularization(model[1].mlp_weight_prompt, 0.01)
                    except:
                        pass
                else:
                    raise ValueError(f"Wrong value for model_name {args.model_name}!")
                # get positive and negative probabilities, shape (batch_size, )
                if args.model_name == "DF" or args.tig_prompt or args.use_dyg_prompt or args.use_ddg_prompt:
                    positive_probabilities = model[2](input_1=batch_src_node_embeddings,
                                                      input_2=batch_dst_node_embeddings).squeeze(dim=-1).sigmoid()
                    negative_probabilities = model[2](input_1=batch_neg_src_node_embeddings,
                                                      input_2=batch_neg_dst_node_embeddings).squeeze(
                        dim=-1).sigmoid()

                else:
                    positive_probabilities = model[1](input_1=batch_src_node_embeddings,
                                                      input_2=batch_dst_node_embeddings).squeeze(dim=-1).sigmoid()
                    negative_probabilities = model[1](input_1=batch_neg_src_node_embeddings,
                                                      input_2=batch_neg_dst_node_embeddings).squeeze(
                        dim=-1).sigmoid()

                predicts = torch.cat([positive_probabilities, negative_probabilities], dim=0)
                labels = torch.cat([torch.ones_like(positive_probabilities), torch.zeros_like(negative_probabilities)], dim=0)

                loss = loss_func(input=predicts, target=labels)
                if l2_loss != None:
                    loss += l2_loss

                train_losses.append(loss.item())

                train_metrics.append(get_link_prediction_metrics(predicts=predicts, labels=labels))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_idx_data_loader_tqdm.set_description(f'Epoch: {epoch + 1}, train for the {batch_idx + 1}-th batch, train loss: {loss.item()}')

                if args.model_name in ['JODIE', 'DyRep', 'TGN']:
                    # detach the memories and raw messages of nodes in the memory bank after each batch, so we don't back propagate to the start of time
                    model[0].memory_bank.detach_memory_bank()

            if args.model_name in ['JODIE', 'DyRep', 'TGN']:
                # backup memory bank after training so it can be used for new validation nodes
                train_backup_memory_bank = model[0].memory_bank.backup_memory_bank()

            val_losses, val_metrics = evaluate_model_link_prediction(model_name=args.model_name,
                                                                     model=model,
                                                                     neighbor_sampler=full_neighbor_sampler,
                                                                     evaluate_idx_data_loader=val_idx_data_loader,
                                                                     evaluate_neg_edge_sampler=val_neg_edge_sampler,
                                                                     evaluate_data=val_data,
                                                                     loss_func=loss_func,
                                                                     num_neighbors=args.num_neighbors,
                                                                     time_gap=args.time_gap,
                                                                     use_tig_prompt=args.tig_prompt,
                                                                     use_dyg_prompt=args.use_dyg_prompt,
                                                                     use_ddg_prompt=args.use_ddg_prompt)

            if args.model_name in ['JODIE', 'DyRep', 'TGN']:
                # backup memory bank after validating so it can be used for testing nodes (since test edges are strictly later in time than validation edges)
                val_backup_memory_bank = model[0].memory_bank.backup_memory_bank()

                # reload training memory bank for new validation nodes
                model[0].memory_bank.reload_memory_bank(train_backup_memory_bank)

            new_node_val_losses, new_node_val_metrics = evaluate_model_link_prediction(model_name=args.model_name,
                                                                                       model=model,
                                                                                       neighbor_sampler=full_neighbor_sampler,
                                                                                       evaluate_idx_data_loader=new_node_val_idx_data_loader,
                                                                                       evaluate_neg_edge_sampler=new_node_val_neg_edge_sampler,
                                                                                       evaluate_data=new_node_val_data,
                                                                                       loss_func=loss_func,
                                                                                       num_neighbors=args.num_neighbors,
                                                                                       time_gap=args.time_gap,
                                                                                       use_tig_prompt=args.tig_prompt,
                                                                                       use_dyg_prompt=args.use_dyg_prompt,
                                                                                       use_ddg_prompt=args.use_ddg_prompt)

            if args.model_name in ['JODIE', 'DyRep', 'TGN']:
                # reload validation memory bank for testing nodes or saving models
                # note that since model treats memory as parameters, we need to reload the memory to val_backup_memory_bank for saving models
                model[0].memory_bank.reload_memory_bank(val_backup_memory_bank)

            logger.info(f'Epoch: {epoch + 1}, learning rate: {optimizer.param_groups[0]["lr"]}, train loss: {np.mean(train_losses):.4f}')
            for metric_name in train_metrics[0].keys():
                logger.info(f'train {metric_name}, {np.mean([train_metric[metric_name] for train_metric in train_metrics]):.4f}')
            logger.info(f'validate loss: {np.mean(val_losses):.4f}')
            for metric_name in val_metrics[0].keys():
                logger.info(f'validate {metric_name}, {np.mean([val_metric[metric_name] for val_metric in val_metrics]):.4f}')
            logger.info(f'new node validate loss: {np.mean(new_node_val_losses):.4f}')
            for metric_name in new_node_val_metrics[0].keys():
                logger.info(f'new node validate {metric_name}, {np.mean([new_node_val_metric[metric_name] for new_node_val_metric in new_node_val_metrics]):.4f}')

            # perform testing once after test_interval_epochs
            if (epoch + 1) % args.test_interval_epochs == 0:
                test_losses, test_metrics = evaluate_model_link_prediction(model_name=args.model_name,
                                                                           model=model,
                                                                           neighbor_sampler=full_neighbor_sampler,
                                                                           evaluate_idx_data_loader=test_idx_data_loader,
                                                                           evaluate_neg_edge_sampler=test_neg_edge_sampler,
                                                                           evaluate_data=test_data,
                                                                           loss_func=loss_func,
                                                                           num_neighbors=args.num_neighbors,
                                                                           time_gap=args.time_gap,
                                                                           use_tig_prompt=args.tig_prompt,
                                                                           use_dyg_prompt=args.use_dyg_prompt,
                                                                           use_ddg_prompt=args.use_ddg_prompt)

                if args.model_name in ['JODIE', 'DyRep', 'TGN']:
                    # reload validation memory bank for new testing nodes
                    model[0].memory_bank.reload_memory_bank(val_backup_memory_bank)

                new_node_test_losses, new_node_test_metrics = evaluate_model_link_prediction(model_name=args.model_name,
                                                                                             model=model,
                                                                                             neighbor_sampler=full_neighbor_sampler,
                                                                                             evaluate_idx_data_loader=new_node_test_idx_data_loader,
                                                                                             evaluate_neg_edge_sampler=new_node_test_neg_edge_sampler,
                                                                                             evaluate_data=new_node_test_data,
                                                                                             loss_func=loss_func,
                                                                                             num_neighbors=args.num_neighbors,
                                                                                             time_gap=args.time_gap,
                                                                                             use_tig_prompt=args.tig_prompt,
                                                                                             use_dyg_prompt=args.use_dyg_prompt,
                                                                                             use_ddg_prompt=args.use_ddg_prompt)

                if args.model_name in ['JODIE', 'DyRep', 'TGN']:
                    # reload validation memory bank for testing nodes or saving models
                    # note that since model treats memory as parameters, we need to reload the memory to val_backup_memory_bank for saving models
                    model[0].memory_bank.reload_memory_bank(val_backup_memory_bank)

                logger.info(f'test loss: {np.mean(test_losses):.4f}')
                for metric_name in test_metrics[0].keys():
                    logger.info(f'test {metric_name}, {np.mean([test_metric[metric_name] for test_metric in test_metrics]):.4f}')
                logger.info(f'new node test loss: {np.mean(new_node_test_losses):.4f}')
                for metric_name in new_node_test_metrics[0].keys():
                    logger.info(f'new node test {metric_name}, {np.mean([new_node_test_metric[metric_name] for new_node_test_metric in new_node_test_metrics]):.4f}')

            # select the best model based on all the validate metrics
            val_metric_indicator = []
            for metric_name in val_metrics[0].keys():
                val_metric_indicator.append((metric_name, np.mean([val_metric[metric_name] for val_metric in val_metrics]), True))
            early_stop = early_stopping.step(val_metric_indicator, model)

            if early_stop:
                break

        # load the best model
        early_stopping.load_checkpoint(model)

        # evaluate the best model
        logger.info(f'get final performance on dataset {args.dataset_name}...')

        # the saved best model of memory-based models cannot perform validation since the stored memory has been updated by validation data
        if args.model_name not in ['JODIE', 'DyRep', 'TGN']:
            val_losses, val_metrics = evaluate_model_link_prediction(model_name=args.model_name,
                                                                     model=model,
                                                                     neighbor_sampler=full_neighbor_sampler,
                                                                     evaluate_idx_data_loader=val_idx_data_loader,
                                                                     evaluate_neg_edge_sampler=val_neg_edge_sampler,
                                                                     evaluate_data=val_data,
                                                                     loss_func=loss_func,
                                                                     num_neighbors=args.num_neighbors,
                                                                     time_gap=args.time_gap,
                                                                     use_tig_prompt=args.tig_prompt,
                                                                     use_dyg_prompt=args.use_dyg_prompt,
                                                                     use_ddg_prompt=args.use_ddg_prompt)

            new_node_val_losses, new_node_val_metrics = evaluate_model_link_prediction(model_name=args.model_name,
                                                                                       model=model,
                                                                                       neighbor_sampler=full_neighbor_sampler,
                                                                                       evaluate_idx_data_loader=new_node_val_idx_data_loader,
                                                                                       evaluate_neg_edge_sampler=new_node_val_neg_edge_sampler,
                                                                                       evaluate_data=new_node_val_data,
                                                                                       loss_func=loss_func,
                                                                                       num_neighbors=args.num_neighbors,
                                                                                       time_gap=args.time_gap,
                                                                                       use_tig_prompt=args.tig_prompt,
                                                                                       use_dyg_prompt=args.use_dyg_prompt,
                                                                                       use_ddg_prompt=args.use_ddg_prompt)

        if args.model_name in ['JODIE', 'DyRep', 'TGN']:
            # the memory in the best model has seen the validation edges, we need to backup the memory for new testing nodes
            val_backup_memory_bank = model[0].memory_bank.backup_memory_bank()

        # 测试
        test_losses, test_metrics = evaluate_model_link_prediction(model_name=args.model_name,
                                                                   model=model,
                                                                   neighbor_sampler=full_neighbor_sampler,
                                                                   evaluate_idx_data_loader=test_idx_data_loader,
                                                                   evaluate_neg_edge_sampler=test_neg_edge_sampler,
                                                                   evaluate_data=test_data,
                                                                   loss_func=loss_func,
                                                                   num_neighbors=args.num_neighbors,
                                                                   time_gap=args.time_gap,
                                                                   use_tig_prompt=args.tig_prompt,
                                                                   use_dyg_prompt=args.use_dyg_prompt,
                                                                   use_ddg_prompt=args.use_ddg_prompt)

        if args.model_name in ['JODIE', 'DyRep', 'TGN']:
            # reload validation memory bank for new testing nodes
            model[0].memory_bank.reload_memory_bank(val_backup_memory_bank)

        new_node_test_losses, new_node_test_metrics = evaluate_model_link_prediction(model_name=args.model_name,
                                                                                     model=model,
                                                                                     neighbor_sampler=full_neighbor_sampler,
                                                                                     evaluate_idx_data_loader=new_node_test_idx_data_loader,
                                                                                     evaluate_neg_edge_sampler=new_node_test_neg_edge_sampler,
                                                                                     evaluate_data=new_node_test_data,
                                                                                     loss_func=loss_func,
                                                                                     num_neighbors=args.num_neighbors,
                                                                                     time_gap=args.time_gap,
                                                                                     use_tig_prompt=args.tig_prompt,
                                                                                     use_dyg_prompt=args.use_dyg_prompt,
                                                                                     use_ddg_prompt=args.use_ddg_prompt)
        # store the evaluation metrics at the current run
        val_metric_dict, new_node_val_metric_dict, test_metric_dict, new_node_test_metric_dict = {}, {}, {}, {}

        if args.model_name not in ['JODIE', 'DyRep', 'TGN']:
            logger.info(f'validate loss: {np.mean(val_losses):.4f}')
            for metric_name in val_metrics[0].keys():
                average_val_metric = np.mean([val_metric[metric_name] for val_metric in val_metrics])
                logger.info(f'validate {metric_name}, {average_val_metric:.4f}')
                val_metric_dict[metric_name] = average_val_metric

            logger.info(f'new node validate loss: {np.mean(new_node_val_losses):.4f}')
            for metric_name in new_node_val_metrics[0].keys():
                average_new_node_val_metric = np.mean([new_node_val_metric[metric_name] for new_node_val_metric in new_node_val_metrics])
                logger.info(f'new node validate {metric_name}, {average_new_node_val_metric:.4f}')
                new_node_val_metric_dict[metric_name] = average_new_node_val_metric

        logger.info(f'test loss: {np.mean(test_losses):.4f}')
        for metric_name in test_metrics[0].keys():
            average_test_metric = np.mean([test_metric[metric_name] for test_metric in test_metrics])
            logger.info(f'test {metric_name}, {average_test_metric:.4f}')
            test_metric_dict[metric_name] = average_test_metric

        logger.info(f'new node test loss: {np.mean(new_node_test_losses):.4f}')
        for metric_name in new_node_test_metrics[0].keys():
            average_new_node_test_metric = np.mean([new_node_test_metric[metric_name] for new_node_test_metric in new_node_test_metrics])
            logger.info(f'new node test {metric_name}, {average_new_node_test_metric:.4f}')
            new_node_test_metric_dict[metric_name] = average_new_node_test_metric

        single_run_time = time.time() - run_start_time
        logger.info(f'Run {run + 1} cost {single_run_time:.2f} seconds.')

        all_train_cost_time = all_train_cost_time + single_run_time

        if args.model_name not in ['JODIE', 'DyRep', 'TGN']:
            val_metric_all_runs.append(val_metric_dict)
            new_node_val_metric_all_runs.append(new_node_val_metric_dict)
        test_metric_all_runs.append(test_metric_dict)
        new_node_test_metric_all_runs.append(new_node_test_metric_dict)

        # avoid the overlap of logs
        if run < args.num_runs - 1:
            logger.removeHandler(fh)
            logger.removeHandler(ch)

        # save model result
        if args.model_name not in ['JODIE', 'DyRep', 'TGN']:
            result_json = {
                "validate metrics": {metric_name: f'{val_metric_dict[metric_name]:.4f}' for metric_name in val_metric_dict},
                "new node validate metrics": {metric_name: f'{new_node_val_metric_dict[metric_name]:.4f}' for metric_name in new_node_val_metric_dict},
                "test metrics": {metric_name: f'{test_metric_dict[metric_name]:.4f}' for metric_name in test_metric_dict},
                "new node test metrics": {metric_name: f'{new_node_test_metric_dict[metric_name]:.4f}' for metric_name in new_node_test_metric_dict}
            }
        else:
            result_json = {
                "test metrics": {metric_name: f'{test_metric_dict[metric_name]:.4f}' for metric_name in test_metric_dict},
                "new node test metrics": {metric_name: f'{new_node_test_metric_dict[metric_name]:.4f}' for metric_name in new_node_test_metric_dict}
            }
        result_json = json.dumps(result_json, indent=4)

        save_result_folder = f"./saved_results/{args.model_name}/{args.dataset_name}"
        os.makedirs(save_result_folder, exist_ok=True)
        save_result_path = os.path.join(save_result_folder, f"{args.save_model_name}.json")

        with open(save_result_path, 'w') as file:
            file.write(result_json)


    print(f"5 runs cost time:-----{all_train_cost_time:.2f} seconds,average{all_train_cost_time / 5:.2f} seconds-----")

    # store the average metrics at the log of the last run
    logger.info(f'metrics over {args.num_runs} runs:')

    if args.model_name not in ['JODIE', 'DyRep', 'TGN']:
        for metric_name in val_metric_all_runs[0].keys():
            logger.info(f'validate {metric_name}, {[val_metric_single_run[metric_name] for val_metric_single_run in val_metric_all_runs]}')
            logger.info(f'average validate {metric_name}, {np.mean([val_metric_single_run[metric_name] for val_metric_single_run in val_metric_all_runs]):.4f} '
                        f'± {np.std([val_metric_single_run[metric_name] for val_metric_single_run in val_metric_all_runs], ddof=1):.4f}')

        for metric_name in new_node_val_metric_all_runs[0].keys():
            logger.info(f'new node validate {metric_name}, {[new_node_val_metric_single_run[metric_name] for new_node_val_metric_single_run in new_node_val_metric_all_runs]}')
            logger.info(f'average new node validate {metric_name}, {np.mean([new_node_val_metric_single_run[metric_name] for new_node_val_metric_single_run in new_node_val_metric_all_runs]):.4f} '
                        f'± {np.std([new_node_val_metric_single_run[metric_name] for new_node_val_metric_single_run in new_node_val_metric_all_runs], ddof=1):.4f}')

    for metric_name in test_metric_all_runs[0].keys():
        logger.info(f'test {metric_name}, {[test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs]}')
        logger.info(f'average test {metric_name}, {np.mean([test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs]):.4f} '
                    f'± {np.std([test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs], ddof=1):.4f}')

    for metric_name in new_node_test_metric_all_runs[0].keys():
        logger.info(f'new node test {metric_name}, {[new_node_test_metric_single_run[metric_name] for new_node_test_metric_single_run in new_node_test_metric_all_runs]}')
        logger.info(f'average new node test {metric_name}, {np.mean([new_node_test_metric_single_run[metric_name] for new_node_test_metric_single_run in new_node_test_metric_all_runs]):.4f} '
                    f'± {np.std([new_node_test_metric_single_run[metric_name] for new_node_test_metric_single_run in new_node_test_metric_all_runs], ddof=1):.4f}')

    sys.exit()
