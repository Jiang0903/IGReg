from math import pi, log
from functools import wraps
import numpy as np
import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat

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
            context = kwargs['image_public']
            normed_context = self.norm_context(context)
            kwargs.update(image_public = normed_context)
            context = kwargs['image_private']
            normed_context = self.norm_context(context)
            kwargs.update(image_private = normed_context)

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
        self.scale = dim_head ** -0.5 #根号下Ct分之一
        self.heads = heads
        #                       256       128
        self.to_q = nn.Linear(query_dim, inner_dim, bias = False)
        #                       128          128
        self.to_k = nn.Linear(context_dim, context_dim, bias = False)
        #                       128          128
        self.to_v = nn.Linear(context_dim, context_dim, bias=False)
        #                          128        256
        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, point_public, image_public, image_private):
        q = self.to_q(point_public)
        k = self.to_k(image_public)
        v = self.to_v(image_private)
        attention_scores = torch.einsum('bnc,bmc->bnm', q, k) * self.scale
        attention_scores = F.softmax(attention_scores, dim=-1)
        out = torch.matmul(attention_scores, v)
        return self.to_out(out) # 最后一步一个 MLP, 128->256

# main class
class AttentionFusion(nn.Module):
    def __init__(
        self,
        image_num,
        dim,                                    # Q dim  128
        latent_dim = 512,                       # Content dim  256
        cross_heads = 1,                        # Cross-Attention Head
        latent_heads = 8,                       # Self-Attention Head
        cross_dim_head = 64,                    # Cross-Attention Head dim  128
        latent_dim_head = 64,                   # Self-Attention Head dim
    ):
        super().__init__() 

        self.image_num = image_num
        self.dim = dim

        self.K = 64
        
        self.pool_public = nn.AvgPool2d((64, 1))
        self.pool_private = nn.AvgPool2d((64, 1))
        self.private_P = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ReLU()
        )
        self.public = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ReLU()
        )
        self.private_I = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU()
        )
        self.fuse_public = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )
        self.fuse_private = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )  
        # Cross-Attention
        self.cross_attend_blocks = nn.ModuleList([
            #           256                 256       128          1                       128                      128                      
            PreNorm(latent_dim, Attention(latent_dim, dim, heads = cross_heads, dim_head = cross_dim_head), context_dim = dim),
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
        point_2dfeature[inds3d[:, 0], inds3d[:, 1], :] = ifeats[inds2d[:, 0], inds2d[:, 1]]
        
        return point_2dfeature


    def forward(
        self,
        image_feats,                    # image feature (120, 160, 256)
        point_feats,                    # point cloud feature(N, 256)
        mask,
        inds2d,
        inds3d
    ):
        x = point_feats  #256

        point_private = self.private_P(x).squeeze(0)
        point_public = self.public(x).squeeze(0)

        if self.image_num == 1:
            image_public = self.public(image_feats)
            image_private = self.private_I(image_feats)

            point_2dfeature_public = torch.ones(mask.size(0), mask.size(1), image_feats.size(2)).cuda()
            point_2dfeature_public[inds3d[:, 0], inds3d[:, 1], :] = image_public[inds2d[:, 0], inds2d[:, 1]]

            point_2dfeature_private = torch.ones(mask.size(0), mask.size(1), image_feats.size(2)).cuda()
            point_2dfeature_private[inds3d[:, 0], inds3d[:, 1], :] = image_private[inds2d[:, 0], inds2d[:, 1]]

        elif self.image_num == 2:
            image_public_1 = self.public(image_feats['image_feats_1'])
            image_public_2 = self.public(image_feats['image_feats_1'])

            image_private_1 = self.private_I(image_feats['image_feats_1'])
            image_private_2 = self.private_I(image_feats['image_feats_2'])

            point_2dfeature_public = torch.ones(point_feats.size(1), self.K, self.dim).cuda()
            point_2dfeature_public = self.get_point_2dfeature(point_2dfeature_public, inds2d['inds2d_1'], inds3d['inds3d_1'], image_public_1)
            point_2dfeature_public = self.get_point_2dfeature(point_2dfeature_public, inds2d['inds2d_2'], inds3d['inds3d_2'], image_public_2)

            point_2dfeature_private = torch.ones(point_feats.size(1), self.K, self.dim).cuda()
            point_2dfeature_private = self.get_point_2dfeature(point_2dfeature_private, inds2d['inds2d_1'], inds3d['inds3d_1'], image_private_1)
            point_2dfeature_private = self.get_point_2dfeature(point_2dfeature_private, inds2d['inds2d_2'], inds3d['inds3d_2'], image_private_2)


        image_public = self.pool_public(point_2dfeature_public).squeeze()
        image_private = self.pool_private(point_2dfeature_private).squeeze()

        # ---- Cross-Attention----
        point_public = point_public.unsqueeze(1)
        cross_attn, cross_ff = self.cross_attend_blocks
                                   # (N, 1, 256)   # (N, 64, 128)
        point_public1 = cross_attn(point_public, image_public = point_2dfeature_public, image_private=point_2dfeature_private) +  point_public
        point_fused = cross_ff(point_public1) + point_public1
        # ---- Cross-Attention----

        point_public = point_public.squeeze(1)
        point_fused = point_fused.squeeze(1)
        point_fused_public = torch.cat((point_public,point_fused),-1)
        point_fused_public = self.fuse_public(point_fused_public)

        point_feature = torch.cat((point_private,point_fused_public),-1)
        point_feature = self.fuse_private(point_feature)

        modal_dict = {'point_public': point_public,
                     'point_private': point_private,
                     'image_public': image_public,
                     'image_private': image_private}

        return point_feature, modal_dict