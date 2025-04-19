import math
import logging
from functools import partial
from collections import OrderedDict
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from .head import MLP

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, return_attention=False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        if return_attention:
            return x, attn
        return x

class GCNLayer(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, adj_matrix):
        """
        :param x: Input node features. Shape: (B, N, D)
        :param adj_matrix: Adjacency matrix (B, N, N)  (technically, attention weights between T and S)
        :return: Output node features. Shape: (B, N, D_out)
        """
        # Support is the result of multiplying features by weights
        support = self.linear(x) # (B,N,D)
        # Output is the result of multiplying adjacency matrix by the support
        output = torch.bmm(adj_matrix, support) # (B, N, N) * (B, N, D) --> (B, N, D)

        return output


class CEBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 keep_ratio_search=1.0, gnn_layers_per_stage = 1, gnn_type="GCN"): # ADDED keep_ratio_search, gnn_layers
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.keep_ratio_search = keep_ratio_search
        self.gnn_type = gnn_type

        # Dynamically create gnn_layers based on gnn_layers_per_stage.  Handles 0 layers gracefully.
        self.gnn_layers = nn.ModuleList()
        for _ in range(gnn_layers_per_stage):
            if self.gnn_type == 'GCN':
                self.gnn_layers.append(GCNLayer(dim, dim))  # Input and output dim are the same
            elif self.gnn_type == 'GAT':
                # Placeholder for GAT layer.  Needs proper GATLayer implementation.
                # Assumes single-head for simplicity;  adjust num_heads as needed.
                raise NotImplementedError("GAT not yet implemented.")
            else:
                raise ValueError(f"Unknown GNN type: {gnn_type}")

        self.norm3 = norm_layer(dim) # Added normalization after the GCN


    def forward(self, x, global_index_t, global_index_s, mask_x=None, ce_template_mask=None,
                ce_keep_rate=None,):

        B, N, C = x.shape
        x_attn, attn = self.attn(self.norm1(x), True)
        x = x + self.drop_path(x_attn)
        x_res = x
        x = self.norm2(x)
        x = self.mlp(x)

        x = x_res + self.drop_path(x)

        lens_t = global_index_t.shape[1]
        lens_s = global_index_s.shape[1]

        #  GNN Processing
        if len(self.gnn_layers) > 0:

            # Extract attention weights (Wzx) and normalize
            w_ts = attn[:, :, :lens_t, lens_t:]  # (B, H, T, S)
            w_ts = w_ts.sum(dim=1) / self.attn.num_heads # Average over heads. (B, T, S)
            w_ts = F.softmax(w_ts, dim=2)

            zeros_tt = torch.zeros(B, lens_t, lens_t, device=x.device)
            zeros_ss = torch.zeros(B, lens_s, lens_s, device=x.device)
            w_st = w_ts.transpose(1, 2)  # (B, N_s, N_t)

            top = torch.cat([zeros_tt, w_ts], dim=2)  # (B, N_t, N_t + N_s)
            bottom = torch.cat([w_st, zeros_ss], dim=2)  # (B, N_s, N_t + N_s)
            adj = torch.cat([top, bottom], dim=1)  # (B, N_t + N_s, N_t + N_s)

            # GNN Layers
            for gnn_layer in self.gnn_layers:
                 x = gnn_layer(x, adj) # Expects input (B, N, D) and adj (B,S,T)

            # x = torch.cat([x_t, x_s], dim=1)
            x = x + self.drop_path(self.norm3(x)) #Add new normalized residual connection

        removed_index_s = None
        if self.keep_ratio_search < 1:
            score_s = torch.mean(attn[:, :, :lens_t, lens_t:], dim=[1, 2])
            score_s = score_s.reshape(B, -1)

            if ce_keep_rate is not None:
                keep_ratio_search = ce_keep_rate
            else:
                keep_ratio_search = self.keep_ratio_search

            keep_indices = torch.argsort(score_s, dim=1, descending=True)[:, :int(keep_ratio_search * score_s.shape[1])]
            removed_index_s = torch.argsort(score_s, dim=1, descending=True)[:, int(keep_ratio_search * score_s.shape[1]):]
            # update global_index_s
            global_index_s = torch.gather(global_index_s, index=keep_indices, dim=1)

            # mask out tokens
            s_indices_keep = keep_indices + lens_t
            indices_keep = torch.cat((global_index_t.long(), s_indices_keep), dim=1)
            x = torch.gather(x, index=indices_keep.unsqueeze(-1).expand(B, -1, C), dim=1)

        return x, global_index_t, global_index_s, removed_index_s, attn
class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, mask=None):
        x = x + self.drop_path(self.attn(self.norm1(x), mask))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

def candidate_elimination(attn: torch.Tensor, tokens: torch.Tensor, lens_t: int, keep_ratio: float, global_index: torch.Tensor, box_mask_z: torch.Tensor):
    """
    Eliminate potential background candidates for computation reduction and noise cancellation.
    Args:
        attn (torch.Tensor): [B, num_heads, L_t + L_s, L_t + L_s], attention weights
        tokens (torch.Tensor):  [B, L_t + L_s, C], template and search region tokens
        lens_t (int): length of template
        keep_ratio (float): keep ratio of search region tokens (candidates)
        global_index (torch.Tensor): global index of search region tokens
        box_mask_z (torch.Tensor): template mask used to accumulate attention weights

    Returns:
        tokens_new (torch.Tensor): tokens after candidate elimination
        keep_index (torch.Tensor): indices of kept search region tokens
        removed_index (torch.Tensor): indices of removed search region tokens
    """
    lens_s = attn.shape[-1] - lens_t
    bs, hn, _, _ = attn.shape

    lens_keep = math.ceil(keep_ratio * lens_s)
    if lens_keep == lens_s:
        return tokens, global_index, None

    attn_t = attn[:, :, :lens_t, lens_t:]

    if box_mask_z is not None:
        box_mask_z = box_mask_z.unsqueeze(1).unsqueeze(-1).expand(-1, attn_t.shape[1], -1, attn_t.shape[-1])
        # attn_t = attn_t[:, :, box_mask_z, :]
        attn_t = attn_t[box_mask_z]
        attn_t = attn_t.view(bs, hn, -1, lens_s)
        attn_t = attn_t.mean(dim=2).mean(dim=1)  # B, H, L-T, L_s --> B, L_s

        # attn_t = [attn_t[i, :, box_mask_z[i, :], :] for i in range(attn_t.size(0))]
        # attn_t = [attn_t[i].mean(dim=1).mean(dim=0) for i in range(len(attn_t))]
        # attn_t = torch.stack(attn_t, dim=0)
    else:
        attn_t = attn_t.mean(dim=2).mean(dim=1)  # B, H, L-T, L_s --> B, L_s

    # use sort instead of topk, due to the speed issue
    # https://github.com/pytorch/pytorch/issues/22812
    sorted_attn, indices = torch.sort(attn_t, dim=1, descending=True)

    topk_attn, topk_idx = sorted_attn[:, :lens_keep], indices[:, :lens_keep]
    non_topk_attn, non_topk_idx = sorted_attn[:, lens_keep:], indices[:, lens_keep:]

    keep_index = global_index.gather(dim=1, index=topk_idx)
    removed_index = global_index.gather(dim=1, index=non_topk_idx)

    # separate template and search tokens
    tokens_t = tokens[:, :lens_t]
    tokens_s = tokens[:, lens_t:]

    # obtain the attentive and inattentive tokens
    B, L, C = tokens_s.shape
    # topk_idx_ = topk_idx.unsqueeze(-1).expand(B, lens_keep, C)
    attentive_tokens = tokens_s.gather(dim=1, index=topk_idx.unsqueeze(-1).expand(B, -1, C))
    # inattentive_tokens = tokens_s.gather(dim=1, index=non_topk_idx.unsqueeze(-1).expand(B, -1, C))

    # compute the weighted combination of inattentive tokens
    # fused_token = non_topk_attn @ inattentive_tokens

    # concatenate these tokens
    # tokens_new = torch.cat([tokens_t, attentive_tokens, fused_token], dim=0)
    tokens_new = torch.cat([tokens_t, attentive_tokens], dim=1)

    return tokens_new, keep_index, removed_index


