import torch
import torch.nn as nn
import torch.nn.functional as F
from IPython import embed

from geotransformer.modules.ops import point_to_node_partition, index_select, pairwise_distance
from geotransformer.modules.registration import get_node_correspondences
from geotransformer.modules.sinkhorn import LearnableLogOptimalTransport
from geotransformer.modules.geotransformer import (
    GeometricTransformer,
    SuperPointMatching,
    SuperPointTargetGenerator,
    LocalGlobalRegistration,
    InitialTransformer
)

from backbone import KPConvFPN

class GeoTransformer(nn.Module):
    def __init__(self, cfg):
        super(GeoTransformer, self).__init__()
        self.num_points_in_patch = cfg.model.num_points_in_patch
        self.matching_radius = cfg.model.ground_truth_matching_radius
        self.anchor_iteration = cfg.model.anchor_iteration

        self.backbone = KPConvFPN(
            cfg.backbone.input_dim,
            cfg.backbone.output_dim,
            cfg.backbone.init_dim,
            cfg.backbone.kernel_size,
            cfg.backbone.init_radius,
            cfg.backbone.init_sigma,
            cfg.backbone.group_norm,
        )

        self.ini = InitialTransformer(
            cfg.image_num,
            cfg.geotransformer.input_dim,
            cfg.geotransformer.image_dim,
            cfg.geotransformer.hidden_dim,
            cfg.geotransformer.num_heads,
            cfg.geotransformer.sigma_d,
        )
        self.transformer = GeometricTransformer(
            cfg.geotransformer.hidden_dim,
            cfg.geotransformer.num_heads,
            cfg.geotransformer.blocks,
            cfg.geotransformer.sigma_d,
            cfg.geotransformer.sigma_a,
            reduction_a=cfg.geotransformer.reduction_a,
        )
        self.optimal_transport_coarse = LearnableLogOptimalTransport(cfg.model.num_sinkhorn_iterations)

        self.coarse_target = SuperPointTargetGenerator(
            cfg.coarse_matching.num_targets, cfg.coarse_matching.overlap_threshold
        )

        self.coarse_matching = SuperPointMatching(
            cfg.coarse_matching.num_correspondences, cfg.coarse_matching.dual_normalization
        )

        self.fine_matching = LocalGlobalRegistration(
            cfg.fine_matching.topk,
            cfg.fine_matching.acceptance_radius,
            mutual=cfg.fine_matching.mutual,
            confidence_threshold=cfg.fine_matching.confidence_threshold,
            use_dustbin=cfg.fine_matching.use_dustbin,
            use_global_score=cfg.fine_matching.use_global_score,
            correspondence_threshold=cfg.fine_matching.correspondence_threshold,
            correspondence_limit=cfg.fine_matching.correspondence_limit,
            num_refinement_steps=cfg.fine_matching.num_refinement_steps,
        )

        self.optimal_transport = LearnableLogOptimalTransport(cfg.model.num_sinkhorn_iterations)

        self.num_anchor = cfg.coarse_matching.num_anchor
        self.r_nms = cfg.coarse_matching.r_nms
    def forward(self, data_dict):
        output_dict = {}

        # Downsample point clouds
        feats = data_dict['features'].detach()
        transform = data_dict['transform'].detach()

        ref_length_c = data_dict['lengths'][-1][0].item()
        ref_length_f = data_dict['lengths'][1][0].item()
        ref_length = data_dict['lengths'][0][0].item()
        points_c = data_dict['points'][-1].detach()
        points_f = data_dict['points'][1].detach()
        points = data_dict['points'][0].detach()

        ref_points_c = points_c[:ref_length_c]
        src_points_c = points_c[ref_length_c:]
        ref_points_f = points_f[:ref_length_f]
        src_points_f = points_f[ref_length_f:]
        ref_points = points[:ref_length]
        src_points = points[ref_length:]

        output_dict['ref_points_c'] = ref_points_c
        output_dict['src_points_c'] = src_points_c
        output_dict['ref_points_f'] = ref_points_f
        output_dict['src_points_f'] = src_points_f
        output_dict['ref_points'] = ref_points
        output_dict['src_points'] = src_points

        # 1. Generate ground truth node correspondences
        _, ref_node_masks, ref_node_knn_indices, ref_node_knn_masks = point_to_node_partition(
            ref_points_f, ref_points_c, self.num_points_in_patch
        )
        _, src_node_masks, src_node_knn_indices, src_node_knn_masks = point_to_node_partition(
            src_points_f, src_points_c, self.num_points_in_patch
        )

        ref_padded_points_f = torch.cat([ref_points_f, torch.zeros_like(ref_points_f[:1])], dim=0)
        src_padded_points_f = torch.cat([src_points_f, torch.zeros_like(src_points_f[:1])], dim=0)
        ref_node_knn_points = index_select(ref_padded_points_f, ref_node_knn_indices, dim=0)
        src_node_knn_points = index_select(src_padded_points_f, src_node_knn_indices, dim=0)

        gt_node_corr_indices, gt_node_corr_overlaps = get_node_correspondences(
            ref_points_c,
            src_points_c,
            ref_node_knn_points,
            src_node_knn_points,
            transform,
            self.matching_radius,
            ref_masks=ref_node_masks,
            src_masks=src_node_masks,
            ref_knn_masks=ref_node_knn_masks,
            src_knn_masks=src_node_knn_masks,
        )

        output_dict['gt_node_corr_indices'] = gt_node_corr_indices
        output_dict['gt_node_corr_overlaps'] = gt_node_corr_overlaps

        # 2. KPFCNN Encoder
        feats_list = self.backbone(feats, data_dict)
        feats_c = feats_list[-1]
        feats_f = feats_list[0]

        # 3. Conditional Transformer
        ref_feats_c = feats_c[:ref_length_c]
        src_feats_c = feats_c[ref_length_c:]
        ref_feats_c = ref_feats_c.unsqueeze(0)
        src_feats_c = src_feats_c.unsqueeze(0)
        ref_points_c = ref_points_c.unsqueeze(0)
        src_points_c = src_points_c.unsqueeze(0)

        ref_feats_c, src_feats_c = self.ini(ref_points_c, src_points_c, 
                                            ref_feats_c, src_feats_c,
                                            ref_points_f, src_points_f,
                                            ref_node_knn_indices, src_node_knn_indices,
                                            ref_node_knn_masks, src_node_knn_masks,
                                            data_dict)


        for i in range(self.anchor_iteration):
            matching_scores = torch.einsum('bnd,bmd->bnm', ref_feats_c, src_feats_c)  # (P, K, K)
            matching_scores = matching_scores / ref_feats_c.shape[-1] ** 0.5
            matching_scores = self.optimal_transport_coarse(matching_scores, ref_node_masks.unsqueeze(0), src_node_masks.unsqueeze(0))[:,:-1,:-1]
            matching_scores = torch.exp(matching_scores)
            
            #top-k
            # corr_scores, corr_indices = matching_scores.view(-1).topk(k=3, largest=True)
            # ref_anchor_indices = corr_indices // matching_scores.shape[2]
            # src_anchor_indices = corr_indices % matching_scores.shape[2]

            #nms
            ref_anchor_indices, src_anchor_indices = self.nms(matching_scores, ref_points_c[0], src_points_c[0])
            corr_scores = matching_scores[0, ref_anchor_indices, src_anchor_indices]

            # output_dict['ref_anchor_corr_' + str(i) + '_indices'] = ref_anchor_indices
            # output_dict['src_anchor_corr_' + str(i) + '_indices'] = src_anchor_indices
            
            # ref_feats_c, src_feats_c, attention_scores = self.transformer(
            #     ref_points_c,
            #     src_points_c,
            #     ref_feats_c,
            #     src_feats_c,
            #     ref_anchor_indices,
            #     src_anchor_indices,
            #     corr_scores,
            # )
            ref_feats_c, src_feats_c = self.transformer(
                ref_points_c,
                src_points_c,
                ref_feats_c,
                src_feats_c,
                ref_anchor_indices,
                src_anchor_indices,
                corr_scores,
            )
        ref_feats_c_norm = F.normalize(ref_feats_c.squeeze(0), p=2, dim=1)
        src_feats_c_norm = F.normalize(src_feats_c.squeeze(0), p=2, dim=1)

        # output_dict['ref_scores_self1'] = attention_scores[0][0]
        # output_dict['ref_scores_self2'] = attention_scores[2][0]
        # output_dict['ref_scores_self3'] = attention_scores[4][0]
        # output_dict['src_scores_self1'] = attention_scores[0][1]
        # output_dict['src_scores_self2'] = attention_scores[2][1]
        # output_dict['src_scores_self3'] = attention_scores[4][1]

        # output_dict['ref_scores_cross1'] = attention_scores[1][0]
        # output_dict['ref_scores_cross2'] = attention_scores[3][0]
        # output_dict['ref_scores_cross3'] = attention_scores[5][0]
        # output_dict['src_scores_cross1'] = attention_scores[1][1]
        # output_dict['src_scores_cross2'] = attention_scores[3][1]
        # output_dict['src_scores_cross3'] = attention_scores[5][1]

        output_dict['ref_feats_c'] = ref_feats_c_norm
        output_dict['src_feats_c'] = src_feats_c_norm

        # 5. Head for fine level matching
        ref_feats_f = feats_f[:ref_length_f]
        src_feats_f = feats_f[ref_length_f:]
        output_dict['ref_feats_f'] = ref_feats_f
        output_dict['src_feats_f'] = src_feats_f

        # 6. Select topk nearest node correspondences
        with torch.no_grad():
            ref_node_corr_indices, src_node_corr_indices, node_corr_scores = self.coarse_matching(
                ref_feats_c_norm, src_feats_c_norm, ref_node_masks, src_node_masks
            )

            output_dict['ref_node_corr_indices'] = ref_node_corr_indices
            output_dict['src_node_corr_indices'] = src_node_corr_indices

            # 7 Random select ground truth node correspondences during training
            if self.training:
                ref_node_corr_indices, src_node_corr_indices, node_corr_scores = self.coarse_target(
                    gt_node_corr_indices, gt_node_corr_overlaps
                )

        # 7.2 Generate batched node points & feats
        ref_node_corr_knn_indices = ref_node_knn_indices[ref_node_corr_indices]  # (P, K)
        src_node_corr_knn_indices = src_node_knn_indices[src_node_corr_indices]  # (P, K)
        ref_node_corr_knn_masks = ref_node_knn_masks[ref_node_corr_indices]  # (P, K)
        src_node_corr_knn_masks = src_node_knn_masks[src_node_corr_indices]  # (P, K)
        ref_node_corr_knn_points = ref_node_knn_points[ref_node_corr_indices]  # (P, K, 3)
        src_node_corr_knn_points = src_node_knn_points[src_node_corr_indices]  # (P, K, 3)

        ref_padded_feats_f = torch.cat([ref_feats_f, torch.zeros_like(ref_feats_f[:1])], dim=0)
        src_padded_feats_f = torch.cat([src_feats_f, torch.zeros_like(src_feats_f[:1])], dim=0)
        ref_node_corr_knn_feats = index_select(ref_padded_feats_f, ref_node_corr_knn_indices, dim=0)  # (P, K, C)
        src_node_corr_knn_feats = index_select(src_padded_feats_f, src_node_corr_knn_indices, dim=0)  # (P, K, C)

        output_dict['ref_node_corr_knn_points'] = ref_node_corr_knn_points
        output_dict['src_node_corr_knn_points'] = src_node_corr_knn_points
        output_dict['ref_node_corr_knn_masks'] = ref_node_corr_knn_masks
        output_dict['src_node_corr_knn_masks'] = src_node_corr_knn_masks

        # 8. Optimal transport
        matching_scores = torch.einsum('bnd,bmd->bnm', ref_node_corr_knn_feats, src_node_corr_knn_feats)  # (P, K, K)
        matching_scores = matching_scores / feats_f.shape[1] ** 0.5
        matching_scores = self.optimal_transport(matching_scores, ref_node_corr_knn_masks, src_node_corr_knn_masks)

        output_dict['matching_scores'] = matching_scores

        # 9. Generate final correspondences during testing
        with torch.no_grad():
            if not self.fine_matching.use_dustbin:
                matching_scores = matching_scores[:, :-1, :-1]

            ref_corr_points, src_corr_points, corr_scores, estimated_transform = self.fine_matching(
                ref_node_corr_knn_points,
                src_node_corr_knn_points,
                ref_node_corr_knn_masks,
                src_node_corr_knn_masks,
                matching_scores,
                node_corr_scores,
            )

            output_dict['ref_corr_points'] = ref_corr_points
            output_dict['src_corr_points'] = src_corr_points
            output_dict['corr_scores'] = corr_scores
            output_dict['estimated_transform'] = estimated_transform

        return output_dict

    def cor_iou(self, ref_points_c, src_points_c, idxs, srcnum):
        ref_idxs = idxs//srcnum
        src_idxs = idxs%srcnum
        ref = ref_points_c[ref_idxs]
        src = src_points_c[src_idxs]
        ref_anchor = ref[-1]
        src_anchor = src[-1]
        ref_dis = torch.norm(ref-ref_anchor, p=2, dim=1)
        src_dis = torch.norm(src-src_anchor, p=2, dim=1)
        dis = torch.stack((ref_dis, src_dis), 1).min(1)[0]
        return dis

    def nms(self, matching_scores, ref_points_c, src_points_c):
        scores = matching_scores.view(-1)
        
        idxs = scores.argsort()
        ref_anchor_indices = []
        src_anchor_indices = []
        
        i=0
        while i<self.num_anchor:
            i = i+1
            max_score_index = idxs[-1]
            ref_index = max_score_index // matching_scores.shape[2]
            src_index = max_score_index % matching_scores.shape[2]
            ref_anchor_indices.append(ref_index)
            src_anchor_indices.append(src_index)
            dis = self.cor_iou(ref_points_c, src_points_c, idxs, matching_scores.shape[2])[:-1]
            idxs = idxs[:-1]
            idxs = idxs[dis>self.r_nms]
        ref_anchor_indices = torch.tensor(ref_anchor_indices, device='cuda')
        src_anchor_indices = torch.tensor(src_anchor_indices, device='cuda')

        return ref_anchor_indices, src_anchor_indices


def create_model(config):
    model = GeoTransformer(config)
    return model


def main():
    from config import make_cfg

    cfg = make_cfg()
    model = create_model(cfg)
    print(model.state_dict().keys())
    print(model)


if __name__ == '__main__':
    main()
