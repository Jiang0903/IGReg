from math import pi, log
from functools import wraps
import numpy as np
import torch
from torch import nn, einsum
import torch.nn.functional as F

# helpers
def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def cache_fn(f):
    cache = None
    @wraps(f)
    def cached_fn(*args, _cache = True, **kwargs):
        if not _cache:
            return f(*args, **kwargs)
        nonlocal cache
        if cache is not None:
            return cache
        cache = f(*args, **kwargs)
        return cache
    return cached_fn

# helper classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim = None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x, **kwargs):
        x = self.norm(x)
        if exists(self.norm_context):
            context = kwargs['image_feats']
            normed_context = self.norm_context(context)
            kwargs.update(image_feats = normed_context)

        return self.fn(x, **kwargs)

class GEGLU(nn.Module):
    def forward(self, x):
        # x ∈ (M4, 2048)
        x, gates = x.chunk(2, dim = -1)
        # using gelu to activate
        return x * F.gelu(gates)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4):
        '''
        dim = 256
        '''
        super().__init__()
        # build Sequential net
        self.net = nn.Sequential(
            #          256  256 * 4 * 2 = 2048
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            #      256 * 4 = 1024  256
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, query_dim, context_dim = None, heads = 8, dim_head = 64):
        '''
        in cross attention:
            query_dim = 256, PC channels
            context_dim = 128, image channels
            heads = 1, cross_attention head
            dim_head = 128, "Ct" in paper, half of PC channels
        '''
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        self.scale = dim_head ** -0.5
        self.heads = heads
        #                       256       256
        self.to_q = nn.Linear(query_dim, inner_dim, bias = False)
        #                       256          256
        self.to_k = nn.Linear(context_dim, context_dim, bias = False)
        #                       256          256
        self.to_v = nn.Linear(context_dim, context_dim, bias=False)
        #                          256        256
        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, point_public, image_feats):
        q = self.to_q(point_public)
        k = self.to_k(image_feats)
        v = self.to_v(image_feats)
        attention_scores = torch.einsum('bnc,bmc->bnm', q, k) * self.scale
        attention_scores = F.softmax(attention_scores, dim=-1)
        out = torch.matmul(attention_scores, v)
        return self.to_out(out)

# main class
class AttentionFusion(nn.Module):
    def __init__(
        self,
        image_num,
        image_dim,       # image dim  256
        latent_dim,      # Content dim  256
        cross_heads,     # Cross-Attention Head
        latent_heads,    # Self-Attention Head
        cross_dim_head,  # Cross-Attention Head dim
        latent_dim_head, # Self-Attention Head dim
    ):
        super().__init__() 

        self.image_num = image_num
        self.dim = image_dim

        self.K = 64
        
        # Cross-Attention
        self.cross_attend_blocks = nn.ModuleList([
            #           256                 256       256          1                       256                      256                      
            PreNorm(latent_dim, Attention(latent_dim, image_dim, heads = cross_heads, dim_head = cross_dim_head), context_dim = image_dim),
            #           256                 256
            PreNorm(latent_dim, FeedForward(latent_dim))
        ])
        get_latent_attn = lambda: PreNorm(latent_dim, Attention(latent_dim, heads = latent_heads, dim_head = latent_dim_head))
        get_latent_ff = lambda: PreNorm(latent_dim, FeedForward(latent_dim))
        get_latent_attn, get_latent_ff = map(cache_fn, (get_latent_attn, get_latent_ff))


    def get_point_2dfeature(
        self,
        point_2dfeature,
        inds2d,
        inds3d,
        ifeats,
    ):
        point_2dfeature[inds3d[:, 0], inds3d[:, 1], :] = ifeats[inds2d[:, 1], inds2d[:, 0]]

        return point_2dfeature


    def forward(
        self,
        image_feats,                    # image feature (120, 160, 256)
        point_feats,                    # point cloud feature(N, 256)
        inds2d,
        inds3d
    ):
        x = point_feats.squeeze()  #256

        if self.image_num == 1:

            point_2dfeature = torch.ones(point_feats.size(1), self.K, image_feats.size(2)).cuda()
            point_2dfeature[inds3d[:, 0], inds3d[:, 1], :] = image_feats[inds2d[:, 1], inds2d[:, 0]]

        elif self.image_num == 2:
            image_feats_1 = image_feats['image_feats_1']
            image_feats_2 = image_feats['image_feats_1']

            point_2dfeature = torch.ones(point_feats.size(1), self.K, self.dim).cuda()
            point_2dfeature = self.get_point_2dfeature(point_2dfeature, inds2d['inds2d_1'], inds3d['inds3d_1'], image_feats_1)
            point_2dfeature = self.get_point_2dfeature(point_2dfeature, inds2d['inds2d_2'], inds3d['inds3d_2'], image_feats_2)

        # ---- Cross-Attention----
        x = x.unsqueeze(1)
        cross_attn, cross_ff = self.cross_attend_blocks
        x = cross_attn(x, image_feats = point_2dfeature) +  x
        fusion_feature = cross_ff(x) + x
        # ---- Cross-Attention----

        return fusion_feature.squeeze()