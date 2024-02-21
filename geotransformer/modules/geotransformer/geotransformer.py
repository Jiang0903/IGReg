import numpy as np
import torch
import torch.nn as nn

from geotransformer.modules.ops import pairwise_distance
from geotransformer.modules.transformer import SinusoidalPositionalEmbedding, RPEConditionalTransformer
from geotransformer.modules.transformer.fusion_2 import AttentionFusion
from geotransformer.modules.transformer.projection import Projection
from geotransformer.modules.geotransformer.Img_Encoder import ImageEncoder
# class GeometricStructureEmbedding(nn.Module):
#     def __init__(self, hidden_dim, sigma_d, sigma_a, reduction_a='max'):
#         super(GeometricStructureEmbedding, self).__init__()
#         self.sigma_d = sigma_d
#         self.sigma_a = sigma_a
#         self.factor_a = 180.0 / (self.sigma_a * np.pi)

#         self.embedding = SinusoidalPositionalEmbedding(hidden_dim)
#         self.proj_d = nn.Linear(hidden_dim, hidden_dim)
#         self.proj_a = nn.Linear(hidden_dim, hidden_dim)

#         self.reduction_a = reduction_a
#         if self.reduction_a not in ['max', 'mean', 'sum']:
#             raise ValueError(f'Unsupported reduction mode: {self.reduction_a}.')


class SelfGeometricStructureEmbedding(nn.Module):
    def __init__(self, hidden_dim, sigma_d):
        super(SelfGeometricStructureEmbedding, self).__init__()
        self.sigma_d = sigma_d
        self.embedding = SinusoidalPositionalEmbedding(hidden_dim)
        self.proj_d = nn.Linear(hidden_dim, hidden_dim)

    @torch.no_grad()
    def get_embedding_indices(self, points):
        r"""Compute the indices of pair-wise distance embedding and triplet-wise angular embedding.

        Args:
            points: torch.Tensor (B, N, 3), input point cloud

        Returns:
            d_indices: torch.FloatTensor (B, N, N), distance embedding indices
            a_indices: torch.FloatTensor (B, N, N, k), angular embedding indices
        """

        dist_map = torch.sqrt(pairwise_distance(points, points))  # (B, N, N)
        d_indices = dist_map / self.sigma_d

        return d_indices

    def forward(self, points):
        d_indices = self.get_embedding_indices(points)

        d_embeddings = self.embedding(d_indices)
        d_embeddings = self.proj_d(d_embeddings)

        return d_embeddings

class CrossGeometricStructureEmbedding(nn.Module):
    def __init__(self, hidden_dim, sigma_d, sigma_a, reduction_a='max'):
        super(CrossGeometricStructureEmbedding, self).__init__()
        self.sigma_d = sigma_d
        self.sigma_a = sigma_a
        self.factor_a = 180.0 / (self.sigma_a * np.pi)

        self.embedding = SinusoidalPositionalEmbedding(hidden_dim)
        self.proj_a = nn.Linear(hidden_dim, hidden_dim)
        self.proj_d = nn.Linear(hidden_dim, hidden_dim)

        self.reduction_a = reduction_a
        if self.reduction_a not in ['max', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {self.reduction_a}.')

    @torch.no_grad()
    def get_embedding_indices(self, points, anchor_points):
        r"""Compute the indices of pair-wise distance embedding and triplet-wise angular embedding.

        Args:
            points: torch.Tensor (B, N, 3), input point cloud
            anchor_points:  torch.Tensor (B, K, 3), input point cloud

        Returns:
            d_indices: torch.FloatTensor (B, N, K), distance embedding indices
            a_indices: torch.FloatTensor (B, N, K), angular embedding indices
        """
        dist_map = torch.sqrt(pairwise_distance(points, anchor_points))  # (B, N, k)
        d_indices = dist_map / self.sigma_d

        ref_vectors = points.unsqueeze(2) - anchor_points.unsqueeze(1)
        # sta_indices = torch.tensor([1, 2, 0]).to('cuda')
        sta_indices = torch.arange(1, anchor_points.shape[1] + 1).to('cuda')
        sta_indices[anchor_points.shape[1] - 1] = 0
        sta_indices = sta_indices[None, None, :,None].repeat(1, ref_vectors.shape[1],1,3)
        anc_vectors = torch.gather(ref_vectors, dim=2, index=sta_indices)

        sin_values = torch.linalg.norm(torch.cross(ref_vectors, anc_vectors, dim=-1), dim=-1)
        cos_values = torch.sum(ref_vectors * anc_vectors, dim=-1)
        angles = torch.atan2(sin_values, cos_values)
        a_indices = angles * self.factor_a

        return d_indices, a_indices

    def forward(self, points, anchor_points, cor_score):
        d_indices, a_indices = self.get_embedding_indices(points, anchor_points)

        a_embeddings = self.embedding(a_indices)
        a_embeddings = self.proj_a(a_embeddings)

        d_embeddings = self.embedding(d_indices)#(B, N, k, d)
        d_embeddings = self.proj_d(d_embeddings)#(B, N, k, d)
        if self.reduction_a == 'max':
            d_embeddings = d_embeddings.max(dim=2)[0] #(B, N, d)
            a_embeddings = a_embeddings.max(dim=2)[0]
        elif self.reduction_a == 'mean':
            d_embeddings = d_embeddings.mean(dim=2) #(B, N, d)
            a_embeddings = a_embeddings.mean(dim=2)
        else:
            d_embeddings = (cor_score[None, None, :,None] * d_embeddings).sum(2) #(B, N, d)
            a_embeddings = (cor_score[None, None, :,None] * a_embeddings).sum(2)

        embeddings = d_embeddings+ a_embeddings
        return embeddings


class GeometricTransformer(nn.Module):
    def __init__(
        self,
        hidden_dim,
        num_heads,
        blocks,
        sigma_d,
        sigma_a,
        dropout=None,
        activation_fn='ReLU',
        reduction_a='max',
    ):
        r"""Geometric Transformer (GeoTransformer).

        Args:
            input_dim: input feature dimension
            output_dim: output feature dimension
            hidden_dim: hidden feature dimension
            num_heads: number of head in transformer
            blocks: list of 'self' or 'cross'
            sigma_d: temperature of distance
            sigma_a: temperature of angles
            angle_k: number of nearest neighbors for angular embedding
            activation_fn: activation function
            reduction_a: reduction mode of angular embedding ['max', 'mean']
        """
        super(GeometricTransformer, self).__init__()

        self.embedding_self = SelfGeometricStructureEmbedding(hidden_dim, sigma_d)
        self.embedding_cross = CrossGeometricStructureEmbedding(hidden_dim, sigma_d, sigma_a, reduction_a=reduction_a)

        # self.transformer = RPEConditionalTransformer(
        #     blocks, hidden_dim, num_heads, dropout=dropout, activation_fn=activation_fn, return_attention_scores=True
        # )
        self.transformer = RPEConditionalTransformer(
            blocks, hidden_dim, num_heads, dropout=dropout, activation_fn=activation_fn
        )
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        ref_points,
        src_points,
        ref_feats,
        src_feats,
        ref_anchor_indices, 
        src_anchor_indices,
        cor_score,
        ref_masks=None,
        src_masks=None,
    ):
        r"""Geometric Transformer

        Args:
            ref_points (Tensor): (B, N, 3)
            src_points (Tensor): (B, M, 3)
            ref_feats (Tensor): (B, N, C)
            src_feats (Tensor): (B, M, C)
            ref_masks (Optional[BoolTensor]): (B, N)
            src_masks (Optional[BoolTensor]): (B, M)

        Returns:
            ref_feats: torch.Tensor (B, N, C)
            src_feats: torch.Tensor (B, M, C)
        """
        ref_anchor_points = ref_points[:, ref_anchor_indices, :]
        src_anchor_points = src_points[:, src_anchor_indices, :]

        ref_embeddings_self = self.embedding_self(ref_points)
        src_embeddings_self = self.embedding_self(src_points)

        ref_embeddings_cross = self.embedding_cross(ref_points, ref_anchor_points, cor_score)
        src_embeddings_cross = self.embedding_cross(src_points, src_anchor_points, cor_score)

        ref_anchor_feats = ref_feats[:, ref_anchor_indices, :]
        src_anchor_feats = src_feats[:, src_anchor_indices, :]

        # ref_feats, src_feats, attention_scores = self.transformer(
        #     ref_feats,
        #     src_feats,
        #     ref_embeddings_self,
        #     src_embeddings_self,
        #     ref_embeddings_cross,
        #     src_embeddings_cross,
        #     masks0=ref_masks,
        #     masks1=src_masks,
        # )
        ref_feats, src_feats = self.transformer(
            ref_feats,
            src_feats,
            ref_embeddings_self,
            src_embeddings_self,
            ref_embeddings_cross,
            src_embeddings_cross,
            masks0=ref_masks,
            masks1=src_masks,
        )

        ref_feats = self.out_proj(ref_feats)
        src_feats = self.out_proj(src_feats)

        # return ref_feats, src_feats, attention_scores
        return ref_feats, src_feats


class InitialTransformer(nn.Module):
    def __init__(
        self,
        image_num,
        input_dim_c,
        input_dim_i,
        hidden_dim,
        num_heads,
        sigma_d_self,
        dropout=None,
        blocks=['self'],
        activation_fn='ReLU'
    ):
        r"""Initial Transformer

        Args:
            input_dim_c: pc input feature dimension
            input_dim_i: im input feature dimension
            hidden_dim: hidden feature dimension
            num_heads: number of head in transformer
            sigma_d_self: temperature of distance
            blocks: list of 'self' or 'cross'
            activation_fn: activation function
        """
        super(InitialTransformer, self).__init__()

        self.image_num = image_num

        self.embedding_self = SelfGeometricStructureEmbedding(hidden_dim, sigma_d_self)

        self.in_proj = nn.Linear(input_dim_c, hidden_dim)

        self.transformer = RPEConditionalTransformer(
            blocks, hidden_dim, num_heads, dropout=dropout, activation_fn=activation_fn
        )
        self.Img_Encoder = ImageEncoder()
        self.fusion_transformer = AttentionFusion(
            image_num=image_num,
            image_dim=input_dim_i,  # the image channels,256
            latent_dim=input_dim_i,  # the PC channels, 256
            cross_heads=1,  # number of heads for cross attention. paper said 1
            latent_heads=8,  # number of heads for latent self attention, 8
            cross_dim_head=input_dim_i,  # number of dimensions per cross attention head
            latent_dim_head=input_dim_i,  # number of dimensions per latent self attention head
        )
        self.out_proj = nn.Linear(hidden_dim * 2, hidden_dim) # 512->256
    
    def project(
        self,
        points_f,
        image,
        intrinsics,
        world2camera,
        node_knn_indices,
        node_knn_masks
    ):
        '''
        Args:
            points_f: (N', 3)
            image: (3, 240, 320)
            intrinsics: (4, 4)
            world2camera: (4, 4)
            node_knn_indices:(N, 64)
        Returns:
            ifeats: (120, 160, 256)
            mask: 
            inds2d: (X, 2) pixel index, X is the number of legal points
            inds3d:(X, 2) the index of the superpoints' neighbors
        '''
        ifeats = self.Img_Encoder(image.unsqueeze(0).cuda()).squeeze(0) # (256, 120, 160)
        ifeats = ifeats.permute(1, 2, 0) # (120, 160, 256)(H,W,C)
        projection = Projection(intrinsics)
        xy_in_image_space, mask = projection.new_projection(points_f, ifeats, world2camera)
        mask = mask[node_knn_indices] & node_knn_masks
        inds3d = torch.nonzero(mask)
        inds2d = xy_in_image_space[node_knn_indices[inds3d[:, 0], inds3d[:, 1]]]

        return ifeats, inds2d, inds3d


    def forward(
        self,
        ref_points,
        src_points,
        ref_feats,
        src_feats,
        ref_points_f, 
        src_points_f,
        ref_node_knn_indices, 
        src_node_knn_indices,
        ref_node_knn_masks, 
        src_node_knn_masks,
        data_dict
    ):
        r"""Geometric Transformer

        Args:
            ref_points (Tensor): (B, N, 3)
            src_points (Tensor): (B, M, 3)
            ref_feats (Tensor): (B, N, C)
            src_feats (Tensor): (B, M, C)
            ref_points_f (Tensor): (N', 3)
            src_points_f (Tensor): (M', 3)
            ref_node_knn_indices (Tensor): (N, K)
            src_node_knn_indices (Tensor): (M, K)
            ref_node_knn_masks (BoolTensor): (N, K)
            src_node_knn_masks (BoolTensor): (M, K)
            data_dict (dictionary): It storages the ref image and src image

        Returns:
            ref_feats: torch.Tensor (B, N, C)
            src_feats: torch.Tensor (B, M, C)
            differ_loss: the loss between public feature and private feature
        """
        ref_embeddings_self = self.embedding_self(ref_points)
        src_embeddings_self = self.embedding_self(src_points)  # 256

        ref_feats = self.in_proj(ref_feats)
        src_feats = self.in_proj(src_feats) # 1024 -> 256

        # get index between 2d and 3d
        if self.image_num == 1:
            ref_ifeats, ref_inds2d, ref_inds3d = self.project(ref_points_f, 
                                                        data_dict['ref_image'],
                                                        data_dict['ref_intrinsics'],
                                                        data_dict['ref_rotation'].float().cuda(),
                                                        ref_node_knn_indices,
                                                        ref_node_knn_masks)
            src_ifeats, src_inds2d, src_inds3d = self.project(src_points_f, 
                                                        data_dict['src_image'],
                                                        data_dict['src_intrinsics'],
                                                        data_dict['src_rotation'].float().cuda(),
                                                        src_node_knn_indices,
                                                        src_node_knn_masks)
            # fusion image and point
            ref_pi_feats = self.fusion_transformer(ref_ifeats, ref_feats, ref_inds2d, ref_inds3d)
            ref_pi_feats = ref_pi_feats.unsqueeze(0)
            src_pi_feats = self.fusion_transformer(src_ifeats, src_feats, src_inds2d, src_inds3d)
            src_pi_feats = src_pi_feats.unsqueeze(0)

        elif self.image_num == 2:
            ref_ifeats_1, ref_inds2d_1, ref_inds3d_1 = self.project(ref_points_f, 
                                                                    data_dict['ref_image_1'],
                                                                    data_dict['ref_intrinsics'],
                                                                    data_dict['ref_world2camera_1'],
                                                                    ref_node_knn_indices,
                                                                    ref_node_knn_masks)

            ref_ifeats_2, ref_inds2d_2, ref_inds3d_2 = self.project(ref_points_f, 
                                                                    data_dict['ref_image_2'],
                                                                    data_dict['ref_intrinsics'],
                                                                    data_dict['ref_world2camera_2'],
                                                                    ref_node_knn_indices,
                                                                    ref_node_knn_masks)
            
            src_ifeats_1, src_inds2d_1, src_inds3d_1 = self.project(src_points_f, 
                                                                    data_dict['src_image_1'],
                                                                    data_dict['src_intrinsics'],
                                                                    data_dict['src_world2camera_1'],
                                                                    src_node_knn_indices,
                                                                    src_node_knn_masks)

            src_ifeats_2, src_inds2d_2, src_inds3d_2 = self.project(src_points_f, 
                                                                    data_dict['src_image_2'],
                                                                    data_dict['src_intrinsics'],
                                                                    data_dict['src_world2camera_2'],
                                                                    src_node_knn_indices,
                                                                    src_node_knn_masks)

            ref_ifeats = {'image_feats_1':ref_ifeats_1, 'image_feats_2':ref_ifeats_2}
            ref_inds2d = {'inds2d_1':ref_inds2d_1, 'inds2d_2':ref_inds2d_2}
            ref_inds3d = {'inds3d_1':ref_inds3d_1, 'inds3d_2':ref_inds3d_2}

            src_ifeats = {'image_feats_1':src_ifeats_1, 'image_feats_2':src_ifeats_2}
            src_inds2d = {'inds2d_1':src_inds2d_1, 'inds2d_2':src_inds2d_2}
            src_inds3d = {'inds3d_1':src_inds3d_1, 'inds3d_2':src_inds3d_2}

            ref_pi_feats = self.fusion_transformer(ref_ifeats, ref_feats, ref_inds2d, ref_inds3d)
            ref_pi_feats = ref_pi_feats.unsqueeze(0)
            src_pi_feats = self.fusion_transformer(src_ifeats, src_feats, src_inds2d, src_inds3d)
            src_pi_feats = src_pi_feats.unsqueeze(0)

        # fusion point-wise distance
        ref_pwd_feats, src_pwd_feats = self.transformer(
            ref_feats,
            src_feats,
            ref_embeddings_self,
            src_embeddings_self,
            embeddings0_cross=None,
            embeddings1_cross=None
        )

        ref_feats = torch.cat((ref_pi_feats, ref_pwd_feats), dim=2)
        src_feats = torch.cat((src_pi_feats, src_pwd_feats), dim=2)

        ref_feats = self.out_proj(ref_feats)
        src_feats = self.out_proj(src_feats)

        return ref_feats, src_feats