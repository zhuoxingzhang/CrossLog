import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GCNConv, GINConv, GATConv, TransformerConv


class MLP(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 num_layers,
                 hidden_units_dim,
                 activation='relu',
                 use_dropout=False,
                 dropout_rate=0.0):
        super().__init__()

        if num_layers <= 0:
            raise ValueError("num_layers must be a non-negative integer")

        if num_layers == 1:
            self.network = nn.Linear(input_dim, output_dim)
            return

        if isinstance(hidden_units_dim, int):
            hidden_dims = [hidden_units_dim] * (num_layers - 1)
        elif isinstance(hidden_units_dim, list):
            if len(hidden_units_dim) != (num_layers - 1):
                raise ValueError(
                    f"hidden_units list length ({len(hidden_units_dim)}) must match num_layers ({num_layers - 1})")
            hidden_dims = hidden_units_dim
        else:
            raise TypeError("hidden_units must be int or list[int]")

        activation_layer = self._get_activation(activation)

        layers = []
        current_dim = input_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, h_dim))

            if activation_layer:
                layers.append(activation_layer)

            if use_dropout:
                layers.append(nn.Dropout(dropout_rate))

            current_dim = h_dim

        layers.append(nn.Linear(current_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def _get_activation(self, activation):
        activations = {
            'relu': nn.ReLU(),
            'sigmoid': nn.Sigmoid(),
            'tanh': nn.Tanh(),
            'leakyrelu': nn.LeakyReLU(),
            'none': None
        }
        if activation.lower() not in activations:
            raise ValueError(f"Unsupported activation: {activation}")
        return activations[activation.lower()]

    def forward(self, x):
        return self.network(x)


class GNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, gnn_type="Transformer", heads=1):
        super(GNN, self).__init__()
        GNN_MAPPING = {
            "GCN": GCNConv,
            "GAT": GATConv,
            "Transformer": TransformerConv,
            "GIN": GINConv,
        }
        if gnn_type not in GNN_MAPPING:
            raise ValueError(f"Unknown GNN type: {gnn_type}. Supported types: {list(GNN_MAPPING.keys())}")
        ConvLayer = GNN_MAPPING[gnn_type]
        self.convs = nn.ModuleList()

        if gnn_type == "Transformer":
            self.convs.append(ConvLayer(in_channels, hidden_channels // heads, heads=heads))
        elif gnn_type == "GIN":
            self.convs.append(ConvLayer(
                Sequential(Linear(in_channels, hidden_channels), ReLU(), Linear(hidden_channels, hidden_channels)),
                train_eps=True))
        else:
            self.convs.append(ConvLayer(in_channels, hidden_channels))

        for _ in range(num_layers - 1):
            if gnn_type == "Transformer":
                self.convs.append(ConvLayer(hidden_channels, hidden_channels // heads, heads=heads))
            elif gnn_type == "GIN":
                self.convs.append(ConvLayer(
                    Sequential(Linear(hidden_channels, hidden_channels), ReLU(),
                               Linear(hidden_channels, hidden_channels)),
                    train_eps=True))
            else:
                self.convs.append(ConvLayer(hidden_channels, hidden_channels))

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
        x = self.convs[-1](x, edge_index)
        return x


class GatedSubgraphNet(nn.Module):
    def __init__(self, hidden_dim, out_dim):
        super().__init__()
        # Scaled Dot-Product Attention 线性映射
        self.W_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_v = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Soft gate
        self.gate_linear = nn.Linear(hidden_dim, hidden_dim)

        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, h, batch):
        d_k = h.size(-1)

        # --------------------
        # Scaled dot-product attention
        # --------------------
        Q = self.W_q(h)  # [N_total, hidden_dim]
        K = self.W_k(h)  # [N_total, hidden_dim]
        V = self.W_v(h)  # [N_total, hidden_dim]

        # attention scores
        scores = torch.matmul(Q, K.T) / (d_k ** 0.5)  # [N_total, N_total]

        # mask 跨图节点
        mask = batch.unsqueeze(0) == batch.unsqueeze(1)
        scores = scores.masked_fill(~mask, float('-inf'))

        # softmax attention weights
        attn_weights = F.softmax(scores, dim=1)  # [N_total, N_total]

        # 子结构表示
        h_sub = torch.matmul(attn_weights, V)  # [N_total, hidden_dim]

        # --------------------
        # 3. Soft gate
        # --------------------
        # gate from contextual node embedding
        gate = torch.sigmoid(self.gate_linear(h))  # [N_total, hidden_dim]
        # gated substructure
        h_sub_gated = gate * h_sub  # [N_total, hidden_dim]

        # --------------------
        # 4. MLP
        # --------------------
        combined = torch.cat([h, h_sub_gated], dim=-1)  # [N_total, hidden_dim*2]
        out = self.mlp(combined)  # [N_total, out_dim]
        return out


def batch_mi_mean(z1, z2):
    """
    Estimate average mutual information for a batch of single-pair samples.

    Args:
        z1: [B, D] tensor, first embeddings of each pair
        z2: [B, D] tensor, second embeddings of each pair

    Returns:
        mi_mean: scalar tensor, average estimated mutual information for the batch
    """
    # normalize embeddings to unit norm
    z1_norm = F.normalize(z1, dim=1)  # [B, D]
    z2_norm = F.normalize(z2, dim=1)  # [B, D]

    # cosine similarity as correlation proxy
    cos_sim = torch.sum(z1_norm * z2_norm, dim=1)  # [B]

    # clamp to avoid numerical issues
    cos_sim = cos_sim.clamp(-0.999, 0.999)

    # Gaussian MI estimate
    mi = -0.5 * torch.log(1 - cos_sim ** 2)  # [B]

    # return batch mean
    mi_mean = mi.mean()

    return mi_mean


def generate_mlp_reduced_dims(input_dim: int, num_hidden_layers: int):
    dims = [input_dim // 2]
    for _ in range(num_hidden_layers - 1):
        next_dim = dims[-1] // 2
        if next_dim < 1:
            raise ValueError("Layer dimension becomes < 1, reduce hidden layers or increase input_dim.")
        dims.append(next_dim)
    return dims


class Pipeline(nn.Module):
    def __init__(self, args):
        super().__init__()
        with open(f"./data/final_towards_target_{args.target_dataset}_event_abstraction_c{args.abs_coupling}.json", "r",
                  encoding="utf-8") as f:
            all_sys_event_cate_dict = json.load(f)
        all_values = list(all_sys_event_cate_dict.values())
        self.abstract_type_list = sorted(set(all_values))

        with open(f"./data/unified_level_event_abstraction_embedding_c{args.abs_coupling}.json", "r",
                  encoding="utf-8") as f:
            all_sys_event_cate_embedding_dict = json.load(f)

        self.num_abstract_types = len(self.abstract_type_list)
        self.num_domain_invar_per_abs = args.num_domain_invar_per_abs
        self.semantic_dim = args.sys_agnostic_feature_dim
        self.hidden_dim = args.hidden_dim
        self.theta_attn = nn.Parameter(torch.tensor(0.0))  # learnable weight for attention fusion

        # -------- 构造 abstract type -> embedding 映射 --------
        # n × semantic_dim
        abstract_type_semantic = []
        for abs_type in self.abstract_type_list:
            if abs_type not in all_sys_event_cate_embedding_dict:
                raise ValueError(f"{abs_type} not found in embedding dict")
            emb = all_sys_event_cate_embedding_dict[abs_type]
            abstract_type_semantic.append(torch.tensor(emb, dtype=torch.float))

        # [n, semantic_dim]
        self.abstract_type_semantic = torch.stack(abstract_type_semantic).to(args.device)
        self.abstract_type_semantic.requires_grad = False
        # 扩展到每个 domain invariant
        # [n, semantic_dim] → [n, 1, semantic_dim] → [n*m, semantic_dim]
        self.domain_invar_semantic_layer = self.abstract_type_semantic.unsqueeze(1) \
            .repeat(1, self.num_domain_invar_per_abs, 1) \
            .view(-1, self.semantic_dim)  # [n*m, semantic_dim]
        self.domain_invar_semantic_layer.requires_grad = False

        self.sys_agnostic_gnn = GNN(args.sys_agnostic_feature_dim, self.hidden_dim, args.gnn_num_layers,
                                    args.gnn_type, args.gnn_num_head)
        self.sys_agnostic_mlp = MLP(
            input_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
            num_layers=args.mlp_num_layers,
            hidden_units_dim=self.hidden_dim,
            activation='relu'
        )
        self.sys_agnostic_gated_subgraph_net = GatedSubgraphNet(hidden_dim=self.hidden_dim, out_dim=self.hidden_dim)
        self.domain_invar_layer = nn.Parameter(torch.randn(self.num_abstract_types * args.num_domain_invar_per_abs, self.hidden_dim))
        self.multihead_attn = nn.MultiheadAttention(self.hidden_dim, 1)
        self.multihead_attn_semantic = nn.MultiheadAttention(self.semantic_dim, 1)
        self.sys_classifier = MLP(
            input_dim=self.hidden_dim,
            output_dim=1,
            num_layers=args.mlp_num_layers,
            hidden_units_dim=generate_mlp_reduced_dims(input_dim=self.hidden_dim, num_hidden_layers=args.mlp_num_layers - 1),
            activation='relu'
        )
        '''freeze above during fine-tuning'''
        '''initialize following parameters during fine-tuning'''
        self.sys_specific_gnn = GNN(args.sys_specific_feature_dim, self.hidden_dim, args.gnn_num_layers,
                                    args.gnn_type, args.gnn_num_head)
        self.sys_specific_mlp = MLP(
            input_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
            num_layers=args.mlp_num_layers,
            hidden_units_dim=self.hidden_dim,
            activation='relu'
        )
        self.anomaly_classifier = MLP(
            input_dim=self.hidden_dim * 2,
            output_dim=1,
            num_layers=args.mlp_num_layers,
            hidden_units_dim=generate_mlp_reduced_dims(input_dim=self.hidden_dim * 2, num_hidden_layers=args.mlp_num_layers - 1),
            activation='relu'
        )

    def forward(self, data):
        x_temp, x_cate, edge_index, batch, x_abs = data.x_temp, data.x_cate, data.edge_index, data.batch, data.abs

        '''system-specific gnn+mlp'''
        h_sys_specific = self.sys_specific_gnn(x_temp, edge_index)
        h_sys_specific = self.sys_specific_mlp(h_sys_specific)

        '''system-agnostic gnn+mlp'''
        h_sys_agnostic = self.sys_agnostic_gnn(x_cate, edge_index)
        h_sys_agnostic = self.sys_agnostic_mlp(h_sys_agnostic)
        '''system-agnostic substructure'''
        h_sys_agnostic = self.sys_agnostic_gated_subgraph_net(h_sys_agnostic, batch)

        '''cross attn to get soft domain invariant'''
        attn_score_rep = self.multihead_attn(h_sys_agnostic, self.domain_invar_layer, self.domain_invar_layer)[1]

        node_semantic = self.abstract_type_semantic[x_abs]  # [N, S]
        attn_score_semantic = self.multihead_attn_semantic(node_semantic, self.domain_invar_semantic_layer, self.domain_invar_semantic_layer)[1]

        theta_attn_nor = torch.sigmoid(self.theta_attn)

        attn_final = theta_attn_nor * attn_score_rep + (1 - theta_attn_nor) * attn_score_semantic

        # =========================================================
        # 用最终 attention 更新 h_sys_agnostic
        # =========================================================
        h_sys_agnostic = torch.matmul(attn_final, self.domain_invar_layer)  # [N, H]

        '''mutual information loss'''
        mi_loss = batch_mi_mean(h_sys_agnostic, h_sys_specific)

        '''concat & anomaly classifier'''
        h_anomaly = torch.cat([h_sys_agnostic, h_sys_specific], dim=-1)
        anomaly_logits = self.anomaly_classifier(h_anomaly).squeeze(-1)

        '''system classifier'''
        sys_logits = self.sys_classifier(F.relu(h_sys_specific)).squeeze(-1)

        return anomaly_logits, sys_logits, mi_loss
