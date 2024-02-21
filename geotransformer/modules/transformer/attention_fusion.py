from math import pi, log
from functools import wraps

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
            context = kwargs['context']
            normed_context = self.norm_context(context)
            kwargs.update(context = normed_context)

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
        #                       128          256
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias = False)
        #                          128        256
        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, x, context = None, mask = None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        temp = self.to_kv(context)
        k,v = temp.chunk(2, dim = -1) # 按照最后一维，也就是256维，将其均匀地一分为二出 K和 V
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))
        # einsum 爱因斯坦求和
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h = h)
            sim.masked_fill_(~mask, max_neg_value)

        attn = sim.softmax(dim = -1)
        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)
        return self.to_out(out) # 最后一步一个 MLP, 128->256

# main class
class AttentionFusion(nn.Module):
    def __init__(
        self,
        depth,                                  # Self-Attention deep
        dim,                                    # Q dim
        latent_dim = 512,                       # Content dim
        cross_heads = 1,                        # Cross-Attention Head
        latent_heads = 8,                       # Self-Attention Head
        cross_dim_head = 64,                    # Cross-Attention Head dim
        latent_dim_head = 64,                   # Self-Attention Head dim
        weight_tie_layers = False,
    ):
        super().__init__()

        # Cross-Attention
        self.cross_attend_blocks = nn.ModuleList([
            #           256                 256       128          1                       128                             128                      
            PreNorm(latent_dim, Attention(latent_dim, dim, heads = cross_heads, dim_head = cross_dim_head), context_dim = dim),
            #\           256                 256
            PreNorm(latent_dim, FeedForward(latent_dim))
        ])
        #
        get_latent_attn = lambda: PreNorm(latent_dim, Attention(latent_dim, heads = latent_heads, dim_head = latent_dim_head))
        get_latent_ff = lambda: PreNorm(latent_dim, FeedForward(latent_dim))
        get_latent_attn, get_latent_ff = map(cache_fn, (get_latent_attn, get_latent_ff))

        self.layers = nn.ModuleList([])
        cache_args = {'_cache': weight_tie_layers}

        # Self-Attention
        for i in range(depth):
            self.layers.append(nn.ModuleList([
                get_latent_attn(**cache_args),
                get_latent_ff(**cache_args)
            ]))

    def forward(
        self,
        data,                           # Content data
        mask = None,                    # mask
        queries_encoder = None,         # Q data
    ):
        # b, *_, device = *data.shape, data.device
        x = queries_encoder

        # ---- Cross-Attention----
        cross_attn, cross_ff = self.cross_attend_blocks
        x = cross_attn(x, context = data, mask = mask) +  x
        x = cross_ff(x) + x
        # ---- Cross-Attention----


        #  ---- Self-Attention ----
        for self_attn, self_ff in self.layers:
            x = self_attn(x) + x
            x = self_ff(x) + x
        #  ---- Self-Attention ----

        return x

