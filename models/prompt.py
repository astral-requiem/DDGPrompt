import numpy as np
from torch.nn import Sequential, Linear, ReLU

import torch
import torch.nn as nn
from torch_geometric.nn.inits import glorot
import torch.nn.functional as F


class Prompt(nn.Module):
    def __init__(self, time_feat_dim: int, channel_embedding_dim: int,num_channels:int,hidden_dim:int,model_name=""):
        super(Prompt, self).__init__()
        if model_name == "gm":
            self.model_name = model_name
            self.channel_embedding_dim = channel_embedding_dim

            self.relu = nn.ReLU()
            # #
            self.time_encoder = TimeEncoder(time_dim=time_feat_dim, parameter_requires_grad=False)
            self.time_bias_layer = nn.Linear(channel_embedding_dim, 1)
            #
            self.edge_weight_layer = nn.Linear(channel_embedding_dim, 1)

            self.mlp_weight_prompt = nn.Sequential(
                nn.Linear(channel_embedding_dim+time_feat_dim, 8),
                nn.ReLU(),
                nn.Linear(8, channel_embedding_dim+time_feat_dim)
            )
        else:
            self.model_name = ""
            self.channel_embedding_dim = channel_embedding_dim

            self.relu = nn.ReLU()
            #
            self.time_encoder = TimeEncoder(time_dim=time_feat_dim, parameter_requires_grad=False)
            self.time_bias_layer = nn.Linear(channel_embedding_dim, 1)
            #
            self.edge_weight_layer = nn.Linear(2 * channel_embedding_dim, 1)

            self.mlp_weight_prompt = nn.Sequential(
                nn.Linear(num_channels * channel_embedding_dim, 8),
                nn.ReLU(),
                nn.Linear(8, num_channels * channel_embedding_dim)
            )


        self.init_emb()

    def set_device(self, device):
        self.device = device
    def init_emb(self):
        for m in self.modules():
            if isinstance(m, Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)



    def add_mlp_weight_prompt(self,patches_data,batch_node_interact_times,neighbor_time,
                              time_projection_layer):

        if self.model_name == "gm":
            a = patches_data.clone()  # bs,len,200

            bs, len, dim = patches_data.size()

            mlp_patches_data = patches_data.view(bs * len, dim)
            mlp_patches_data = self.mlp_weight_prompt(mlp_patches_data)
            mlp_patches_data = mlp_patches_data.view(bs, len, dim)

            edge_weight = self.edge_weight_layer(patches_data[:, :, 0:self.channel_embedding_dim])  # bs,len,1
            edge_weight_patches_data = patches_data * edge_weight
            
            b = patches_data[:, :, 0:self.channel_embedding_dim]
            time_bias = self.time_bias_layer(b).squeeze()  # bs,2len,(1)
            delta_time = torch.from_numpy(batch_node_interact_times[:, np.newaxis] - neighbor_time).float().to(
                self.device)
            delta_time = self.relu(delta_time + time_bias)
            time_mask = (neighbor_time == 0)
            time_fea = self.time_encoder(delta_time)
            time_fea[time_mask] = 0.0  # bs,2len,50
            time_bias_patches_data = patches_data.clone() 
            time_bias_patches_data[:, :,self.channel_embedding_dim:] = time_fea

            return a
        else:
            a = patches_data.clone()        # bs,len,200

            bs,len,dim = patches_data.size()

            mlp_patches_data = patches_data.view(bs*len,dim)
            mlp_patches_data = self.mlp_weight_prompt(mlp_patches_data)
            mlp_patches_data = mlp_patches_data.view(bs,len,dim)

            edge_weight = self.edge_weight_layer(patches_data[:,:,0:2*self.channel_embedding_dim])          # bs,len,1
            edge_weight_patches_data = patches_data * edge_weight

            b = patches_data[:,:,0:self.channel_embedding_dim]
            time_bias = self.time_bias_layer(b).squeeze()      # bs,2len,(1)
            delta_time = torch.from_numpy(batch_node_interact_times[:,np.newaxis] - neighbor_time).float().to(self.device)
            delta_time = self.relu(delta_time + time_bias)
            time_mask = (neighbor_time == 0)
            time_fea = self.time_encoder(delta_time)
            time_fea[time_mask] = 0.0           # bs,2len,50
            time_fea = time_projection_layer(time_fea)
            time_bias_patches_data = patches_data.clone() 
            time_bias_patches_data[:,:,2*self.channel_embedding_dim:3*self.channel_embedding_dim] = time_fea


            return a  +  1 * edge_weight_patches_data  +  1 * mlp_patches_data+ 1 * time_bias_patches_data




    def __init__(self, node_feat_dim: int = 172,time_feat_dim: int = 100,alpha: int = 2):
        super(dyg_prompt, self).__init__()

        # node_feat_dim = 14

        self.feature_prompt = torch.nn.Parameter(torch.Tensor(1, node_feat_dim))
        self.time_prompt = torch.nn.Parameter(torch.Tensor(1, time_feat_dim))

        self.time_con_net = nn.Sequential(
            nn.Linear(time_feat_dim, int(time_feat_dim / alpha)),
            nn.ReLU(),
            nn.Linear(int(time_feat_dim / alpha), node_feat_dim)
        )

        self.feature_con_net = nn.Sequential(
            nn.Linear(node_feat_dim, int(node_feat_dim / alpha)),
            nn.ReLU(),
            nn.Linear(int(node_feat_dim / alpha), time_feat_dim)
        )
        self.reset_parameters()
        self.init_emb()

    def reset_parameters(self):
        glorot(self.feature_prompt)
        glorot(self.time_prompt)
    def init_emb(self):
        for m in self.modules():
            if isinstance(m, Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def set_device(self, device):
        self.device = device

    def add_dyg_prompt(self, node_embeddings, node_time_features):  # (bs,172) (bs,1,100)

        node_time_features = node_time_features.mul(self.time_prompt)

        node_embeddings = node_embeddings.mul(self.feature_prompt)

        feature_con_time_prompt = self.feature_con_net(node_embeddings)

        time_con_feature_prompt = self.time_con_net(node_time_features)

        if feature_con_time_prompt.ndim == 2:
            feature_con_time_prompt = feature_con_time_prompt.unsqueeze(dim=1)
        time_con_feature_prompt = time_con_feature_prompt.squeeze()

        node_time_features = node_time_features.mul(feature_con_time_prompt)

        node_embeddings = node_embeddings.mul(time_con_feature_prompt)

        return node_embeddings, node_time_features

    def add_dyg_prompt_(self,  src_padded_nodes_neighbor_node_raw_features,src_padded_nodes_neighbor_time_features,
                        dst_padded_nodes_neighbor_node_raw_features,dst_padded_nodes_neighbor_time_features,model=""):  # (bs,172) (bs,1,100)

        if model == "gm":
            bs, f_dim = src_padded_nodes_neighbor_node_raw_features.shape  # len1

            _, _, t_dim = src_padded_nodes_neighbor_time_features.shape  # len1


            src_padded_nodes_neighbor_time_features = src_padded_nodes_neighbor_time_features.mul(self.time_prompt)  # bs,len1+len2,dim
            dst_padded_nodes_neighbor_time_features = dst_padded_nodes_neighbor_time_features.mul(self.time_prompt)  # bs,len1+len2,dim

            src_padded_nodes_neighbor_node_raw_features = src_padded_nodes_neighbor_node_raw_features.mul(self.feature_prompt)  # 2bs,dim
            dst_padded_nodes_neighbor_node_raw_features = dst_padded_nodes_neighbor_node_raw_features.mul(self.feature_prompt)  # 2bs,dim


            src_feature_con_time_prompt = self.feature_con_net(src_padded_nodes_neighbor_node_raw_features)   # 2bs,dim
            dst_feature_con_time_prompt = self.feature_con_net(dst_padded_nodes_neighbor_node_raw_features)   # 2bs,dim

            src_time_con_feature_prompt = self.time_con_net(src_padded_nodes_neighbor_time_features)
            dst_time_con_feature_prompt = self.time_con_net(dst_padded_nodes_neighbor_time_features)

            src_padded_nodes_neighbor_time_features = src_padded_nodes_neighbor_time_features.mul(src_feature_con_time_prompt.unsqueeze(dim=1))  # bs,len1+len2,dim
            dst_padded_nodes_neighbor_time_features = dst_padded_nodes_neighbor_time_features.mul(dst_feature_con_time_prompt.unsqueeze(dim=1))  # bs,len1+len2,dim

            src_padded_nodes_neighbor_node_raw_features = src_padded_nodes_neighbor_node_raw_features.mul(torch.mean(src_time_con_feature_prompt,dim=1))  # bs,len1+len2,dim
            dst_padded_nodes_neighbor_node_raw_features = dst_padded_nodes_neighbor_node_raw_features.mul(torch.mean(dst_time_con_feature_prompt,dim=1))  # bs,len1+len2,dim



            return src_padded_nodes_neighbor_node_raw_features, src_padded_nodes_neighbor_time_features, \
                   dst_padded_nodes_neighbor_node_raw_features, dst_padded_nodes_neighbor_time_features

        else:

            bs,src_num,f_dim = src_padded_nodes_neighbor_node_raw_features.shape      # len1
            _,_,t_dim = src_padded_nodes_neighbor_time_features.shape      # len1

            node_raw_features = torch.cat((src_padded_nodes_neighbor_node_raw_features,dst_padded_nodes_neighbor_node_raw_features),dim=1)
            time_features = torch.cat((src_padded_nodes_neighbor_time_features,dst_padded_nodes_neighbor_time_features),dim=1)

            time_features = time_features.mul(self.time_prompt)         # bs,len1+len2,dim
            node_raw_features = node_raw_features.mul(self.feature_prompt)      # bs,len1+len2,dim

            node_raw_features = node_raw_features.view(-1,f_dim)
            feature_con_time_prompt = self.feature_con_net(node_raw_features)
            feature_con_time_prompt = feature_con_time_prompt.view(bs,-1,t_dim)
            node_raw_features = node_raw_features.view(bs,-1,f_dim)

            time_features = time_features.view(-1, t_dim)
            time_con_feature_prompt = self.time_con_net(time_features)
            time_con_feature_prompt = time_con_feature_prompt.view(bs,-1,f_dim)
            time_features = time_features.view(bs, -1, t_dim)




            time_features = time_features.mul(feature_con_time_prompt)  # bs,len1+len2,dim

            node_raw_features = node_raw_features.mul(time_con_feature_prompt)  # bs,len1+len2,dim

            dst_padded_nodes_neighbor_node_raw_features = node_raw_features[:,src_num:,:]
            src_padded_nodes_neighbor_node_raw_features = node_raw_features[:,:src_num,:]

            dst_padded_nodes_neighbor_time_features = time_features[:,src_num:,:]
            src_padded_nodes_neighbor_time_features = time_features[:,:src_num,:]

            return src_padded_nodes_neighbor_node_raw_features, src_padded_nodes_neighbor_time_features,\
            dst_padded_nodes_neighbor_node_raw_features, dst_padded_nodes_neighbor_time_features
        # return node_raw_features, time_features












class TimeEncoder(nn.Module):

    def __init__(self, time_dim: int, parameter_requires_grad: bool = True):
        """
        Time encoder.
        :param time_dim: int, dimension of time encodings
        :param parameter_requires_grad: boolean, whether the parameter in TimeEncoder needs gradient
        """
        super(TimeEncoder, self).__init__()

        self.time_dim = time_dim
        # trainable parameters for time encoding
        self.w = nn.Linear(1, time_dim)
        self.w.weight = nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, time_dim, dtype=np.float32))).reshape(time_dim, -1))
        self.w.bias = nn.Parameter(torch.zeros(time_dim))

        if not parameter_requires_grad:
            self.w.weight.requires_grad = False
            self.w.bias.requires_grad = False

    def forward(self, timestamps: torch.Tensor):
        """
        compute time encodings of time in timestamps
        :param timestamps: Tensor, shape (batch_size, seq_len)
        :return:
        """
        # Tensor, shape (batch_size, seq_len, 1)
        timestamps = timestamps.unsqueeze(dim=-1)

        # Tensor, shape (batch_size, seq_len, time_dim)
        output = torch.cos(self.w(timestamps))

        return output

