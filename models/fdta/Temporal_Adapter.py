import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x, class_token=False):
        x = x.permute(1, 2, 0)

        num_feats = x.shape[1]
        num_pos_feats = num_feats
        mask = torch.zeros(x.shape[0], x.shape[2], device=x.device).to(torch.bool)
        batch = mask.shape[0]
        assert mask is not None
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / num_pos_feats)

        pos_y = y_embed[:, :, None] / dim_t
        pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
        return pos_y


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")



class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask = None,
                     src_key_padding_mask = None,
                     pos = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask = None,
                    src_key_padding_mask = None,
                    pos = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask = None,
                src_key_padding_mask = None,
                pos = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)



class History_motion_embedding(nn.Module):
    def __init__(self, d_model=256, nhead=8, dim_feedforward=512, dropout=0.1,
                 activation='relu', normalize_before=False, pos_type='sin'):
        super(History_motion_embedding, self).__init__()
        self.cascade_num = 6
        self.nhead = nhead
        self.d_model = d_model
        # self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.trca = nn.ModuleList()
        for _ in range(self.cascade_num):
            self.trca.append(TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                   dropout, activation, normalize_before))

        if pos_type == 'sin':
            self.pose_encoding = PositionEmbeddingSine(normalize=True)
    

    def forward(self, seq_info):
        trajectory_features = seq_info['trajectory_features']
        trajectory_masks = seq_info['trajectory_masks']
        batch_size, num_groups, seq_len, num_objs, feature_dim = trajectory_features.shape
    
        # 重塑为 [30, 60, 256] - seq_len, (num_groups * num_objs), feature_dim
        trajectory_features = trajectory_features.squeeze(0)  # [6, 30, 10, 256]
        trajectory_features = trajectory_features.permute(1, 0, 2, 3)  # [30, 6, 10, 256]
        q_patch = trajectory_features.contiguous().view(seq_len, num_groups * num_objs, feature_dim)  # [30, 60, 256]        
         # 同样重塑trajectory_masks
        trajectory_masks = trajectory_masks.squeeze(0)  # [6, 30, 10]
        trajectory_masks = trajectory_masks.permute(1, 0, 2)  # [30, 6, 10]
        trajectory_masks = trajectory_masks.contiguous().view(seq_len, num_groups * num_objs)  # [30, 60]
        pos = self.pose_encoding(q_patch).transpose(0, 1)
        n, b, d = q_patch.shape
        

        # mask missed frames
        missing_frames = trajectory_masks.transpose(0, 1)  # [60, 30] - (num_groups * num_objs, seq_len)
        trajectory_mask = missing_frames[:, None, None].expand(-1, self.nhead, n, -1).reshape(-1, n, n)
        trajectory_mask.diagonal(dim1=1, dim2=2).fill_(False)  # 对角线不遮挡
        # causal_mask 
        causal_mask = (1-torch.tril(torch.ones(n, n, device=q_patch.device))).bool()
        causal_mask = causal_mask.unsqueeze(0).expand(self.nhead * b, -1, -1)
        
        for i in range(self.cascade_num):
            en_out = self.trca[i](src=q_patch, src_mask=causal_mask|trajectory_mask, pos=pos)
            q_patch = en_out
        en_out = en_out.view(1, seq_len, num_groups, num_objs, feature_dim)
        seq_info['trajectory_features'] = en_out.transpose(2,1)  # [1, 6, 30, 10, 256]  
                    
        return seq_info
    

