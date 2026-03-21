# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Deformable DETR model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn
import math
from torch import Tensor  
import warnings

from utils import box_ops
from utils.nested_tensor import NestedTensor, nested_tensor_from_tensor_list
from models.misc import inverse_sigmoid, accuracy, interpolate
from utils.misc import is_distributed, distributed_world_size
# from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
#                        accuracy, get_world_size, interpolate,
#                        is_dist_avail_and_initialized, inverse_sigmoid)

from models.deformable_detr.backbone import build_backbone
from ..Spatial_Adapter.depth_predictor import DepthPredictor
from ..Spatial_Adapter.ddn_loss import DDNLoss
from .matcher import build_matcher
from .segmentation import (DETRsegm, PostProcessPanoptic, PostProcessSegm,
                           dice_loss, sigmoid_focal_loss)
from .deformable_transformer import build_deforamble_transformer
import copy


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class DeformableDETR(nn.Module):
    """ This is the Deformable DETR module that performs object detection """
    def __init__(self, backbone, depth_predictor, transformer, num_classes, num_queries, num_feature_levels,
                 aux_loss=True, with_box_refine=False, two_stage=False):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            with_box_refine: iterative bounding box refinement
            two_stage: two-stage Deformable DETR
        """
        super().__init__()
        self.num_queries = num_queries
        self.depth_predictor = depth_predictor
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.num_feature_levels = num_feature_levels
        if not two_stage:
            self.query_embed = nn.Embedding(num_queries, hidden_dim*2)
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.strides)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[0], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine
        self.two_stage = two_stage

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        # if two-stage, the last class_embed and bbox_embed is for region proposal generation
        num_pred = (transformer.decoder.num_layers + 1) if two_stage else transformer.decoder.num_layers
        if with_box_refine:
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            # hack implementation for iterative bounding box refinement
            self.transformer.decoder.bbox_embed = self.bbox_embed
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            self.transformer.decoder.bbox_embed = None
        if two_stage:
            # hack implementation for two-stage
            self.transformer.decoder.class_embed = self.class_embed
            for box_embed in self.bbox_embed:
                nn.init.constant_(box_embed.layers[-1].bias.data[2:], 0.0)

    def forward(self, samples: NestedTensor, depthmaps: NestedTensor = None):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - depthmaps.tensor: batched depthmaps, of shape [batch_size x 1 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)

        query_embeds = None
        if not self.two_stage:
            query_embeds = self.query_embed.weight
            
        pred_depth_map_logits, depth_pos_embed, weighted_depth, depth_pos_embed_ip = self.depth_predictor([src.detach() for src in srcs], masks[0], pos[0])
        hs, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact = self.transformer(srcs, masks, pos, query_embeds, depth_pos_embed, depth_pos_embed_ip)

        outputs_classes = []
        outputs_coords = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.class_embed[lvl](hs[lvl])
            tmp = self.bbox_embed[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)

        out = {'pred_logits': outputs_class[-1], 
               'pred_boxes': outputs_coord[-1], 
               'pred_depth_map_logits': pred_depth_map_logits, 
               'weighted_depth': weighted_depth, 
               'depthmaps': depthmaps}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)

        if self.two_stage:
            enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
            out['enc_outputs'] = {'pred_logits': enc_outputs_class, 'pred_boxes': enc_outputs_coord}

        # Output the outputs of last decoder layer.
        # We need these outputs to generate the embeddings for objects.
        out["outputs"] = hs[-1]
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, losses, ddn_mode, fg_weight=6, 
                 num_bins=150, focal_alpha=0.25, temperature=0.1):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            ddn_mode: Depth discretization mode
            fg_weight: Foreground weight for DDN loss
            num_bins: Number of depth bins
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha
        self.ddn_loss = DDNLoss(ddn_mode=ddn_mode, fg_weight=fg_weight, num_bins=num_bins)
        self.track_embed = MLP(256, 256, 256, 3)
        # contrastive loss
        self.contrastive_loss = InstanceLevelContrastiveLoss(tau=temperature)

    def loss_labels(self, outputs, targets, indices, indices_iou, num_boxes, epoch, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:,:,:-1]
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, indices_iou, num_boxes, epoch):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, indices_iou, num_boxes, epoch):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_masks(self, outputs, targets, indices, indices_iou, num_boxes, epoch):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)

        src_masks = outputs["pred_masks"]

        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list([t["masks"] for t in targets]).decompose()
        target_masks = target_masks.to(src_masks)

        src_masks = src_masks[src_idx]
        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks[tgt_idx].flatten(1)

        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses
    
    def loss_depth_map(self, outputs, targets, indices, indices_iou, num_boxes, epoch):
        depth_map_logits = outputs['pred_depth_map_logits']
        _, _, h, w = depth_map_logits.shape
        depthmaps = outputs['depthmaps'].tensors

        num_gt_per_img = [len(t['boxes']) for t in targets]
        gt_boxes2d = torch.cat([t['boxes'] for t in targets], dim=0) * torch.tensor([w, h, w, h], device='cuda')
        gt_boxes2d = box_ops.box_cxcywh_to_xyxy(gt_boxes2d)
        
        losses = dict()

        losses["loss_depth_map"] = self.ddn_loss(
            depth_map_logits, gt_boxes2d, num_gt_per_img, depthmaps)
        return losses
    
    def loss_Identity_Adapter(self, outputs, targets, indices, indices_iou, num_boxes, epoch, **kwargs):
        """Compute Identity Adapter loss for instance association within the batch.
        
        Args:
            outputs: Model outputs, using 'outputs' key with shape [num_frames, num_queries, embed_dim]
            targets: List of target annotations (length equals num_frames)
            indices: Hungarian matching results (list of length num_frames)
            indices_iou: IoU matching results per frame (list of length num_frames)
            num_boxes: Total number of target boxes
            epoch: Current training epoch
        
        Returns:
            dict: Dictionary containing 'loss_contrastive' tensor
        """
        if epoch == 0:
            # Skip contrastive loss at epoch 0 to allow the model to warm up
            device = outputs['pred_logits'].device if 'pred_logits' in outputs else targets[0]['labels'].device
            return {'loss_contrastive': torch.tensor(0.0, device=device, requires_grad=True)}
                
        # Collect all matched embeddings and corresponding instance IDs
        all_embeddings = []
        all_instance_ids = []
        all_iou_weights = []
        frame_sources = []
        
        # Iterate through each frame in the sequence
        for frame_idx, ((pred_indices, gt_indices), frame_ious) in enumerate(zip(indices, indices_iou)):
            # Skip frames with no matched predictions
            if len(pred_indices) == 0:
                continue
            
            high_quality_mask = frame_ious >= 0.5
            if high_quality_mask.sum() == 0:
                continue
            filtered_pred_indices = pred_indices[high_quality_mask.cpu()]
            filtered_gt_indices = gt_indices[high_quality_mask.cpu()]
            filtered_iou = frame_ious[high_quality_mask.cpu()]
            
            # Extract matched detection embeddings for this frame
            frame_embeddings = outputs['outputs'][frame_idx, filtered_pred_indices, :]  # [num_matched, embed_dim]
            all_embeddings.append(frame_embeddings)
            all_instance_ids.append(targets[frame_idx]['id'][filtered_gt_indices])
            all_iou_weights.append(filtered_iou)
            frame_sources.extend([frame_idx] * len(filtered_pred_indices))
        
        # Check if sufficient data has been collected
        if len(all_embeddings) == 0:
            warnings.warn("There are no matched embeddings in the batch.")
            device = outputs['pred_logits'].device
            return {'loss_contrastive': torch.tensor(0.0, device=device)}  # Return zero loss early
        
        # Concatenate all embeddings and instance IDs
        embeddings = torch.cat(all_embeddings, dim=0)  # [total_matched, embed_dim]
        instance_ids = torch.cat(all_instance_ids, dim=0)  # [total_matched]
        iou_weights = torch.cat(all_iou_weights, dim=0)  # [total_matched]
        
        # Verify sufficient samples for contrastive learning
        if embeddings.shape[0] < 2:
            warnings.warn("Not enough matched embeddings for contrastive learning.")
            device = outputs['pred_logits'].device
            return {'loss_contrastive': torch.tensor(0.0, device=device)}  # Return zero loss early
    
        # Create contrastive learning labels
        # Positive pairs: detections with the same instance_id across frames
        # Negative pairs: detections with different instance_ids
        contrastive_labels = self._create_simple_labels(instance_ids)
        
        # Compute contrastive loss
        loss_cont = self.contrastive_loss(self.track_embed(embeddings), contrastive_labels, iou_weights)
        
        return {'loss_contrastive': loss_cont}

    def _create_simple_labels(self, instance_ids):
        """Create contrastive learning labels by mapping instance IDs to sequential indices.
        
        Args:
            instance_ids: Tensor of instance IDs with shape [N]
        
        Returns:
            contrastive_labels: Tensor of contrastive labels with shape [N],
                where identical instance_ids are assigned the same label index
        """
        unique_instance_ids = torch.unique(instance_ids)
        contrastive_labels = torch.zeros_like(instance_ids, dtype=torch.long)
        
        # Assign a unique label index to each distinct instance_id
        for i, unique_id in enumerate(unique_instance_ids):
            mask = (instance_ids == unique_id)
            contrastive_labels[mask] = i
        
        return contrastive_labels
    
    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, indices_iou ,num_boxes, **kwargs):
        assert "batch_len" in kwargs, f"batch_len is not in kwargs"
        batch_len = kwargs["batch_len"]
        epoch = kwargs["epoch"]
        kwargs = {}     # to default setting

        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks,
            'depth_map': self.loss_depth_map,
            'contrastive': self.loss_Identity_Adapter
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'

        # Organize the batch data:
        loss_dict = {}
        iter_idxs = torch.tensor(list(range(0, len(targets))), dtype=torch.int64, device=outputs['pred_logits'].device)
        from train import batch_iterator, tensor_dict_index_select
        for batch_iter_idxs, batch_targets, batch_indices, batch_indices_iou in batch_iterator(
            batch_len, iter_idxs, targets, indices, indices_iou
        ):
            batch_outputs = tensor_dict_index_select(outputs, batch_iter_idxs, dim=0)
            batch_loss_dict = loss_map[loss](batch_outputs, batch_targets, batch_indices, batch_indices_iou, 1, epoch, **kwargs)  # num_boxes=1
            for k, v in batch_loss_dict.items():
                if k not in loss_dict:
                    loss_dict[k] = v
                else:
                    loss_dict[k] = loss_dict[k] + v
        # Average the loss:
        if loss == "labels" or loss == "boxes" or loss == "masks" or loss == "depth_map":
            for k in loss_dict.keys():
                loss_dict[k] = loss_dict[k] / num_boxes
        pass
        return loss_dict
        # return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets, **kwargs):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        epoch = kwargs.get('epoch')
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs' and k != 'enc_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        if "batch_len" not in kwargs:
            indices , indices_iou = self.matcher(outputs_without_aux, targets)
        else:
            indices = []
            indices_iou = []
            iter_idxs = torch.tensor(
                list(range(0, len(targets))), dtype=torch.int64, device=outputs_without_aux['pred_logits'].device
            )
            from train import batch_iterator, tensor_dict_index_select
            for batch_iter_idxs, batch_targets in batch_iterator(
                    kwargs["batch_len"], iter_idxs, targets
            ):
                batch_outputs_without_aux = tensor_dict_index_select(outputs_without_aux, batch_iter_idxs, dim=0)
                _ , _iou = self.matcher(batch_outputs_without_aux, batch_targets)
                indices += _
                indices_iou += _iou
                pass

        batch_len = kwargs["batch_len"]         # HELLORPG Added
        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_distributed():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / distributed_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            kwargs = {"batch_len": kwargs["batch_len"], "epoch": epoch}  
            losses.update(self.get_loss(loss, outputs, targets, indices, indices_iou, num_boxes, **kwargs))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices , indices_iou = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    if loss == 'depth_map':
                        continue
                    if loss == 'contrastive':
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs['log'] = False
                    kwargs["batch_len"] = batch_len     # HELLORPG Added
                    kwargs["epoch"] = epoch
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, indices_iou, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if 'enc_outputs' in outputs:
            enc_outputs = outputs['enc_outputs']
            bin_targets = copy.deepcopy(targets)
            for bt in bin_targets:
                bt['labels'] = torch.zeros_like(bt['labels'])
            indices , indices_iou = self.matcher(enc_outputs, bin_targets)
            for loss in self.losses:
                if loss == 'masks':
                    # Intermediate masks losses are too costly to compute, we ignore them.
                    continue
                if loss == 'depth_map':
                    continue
                if loss == 'contrastive':
                    continue
                kwargs = {}
                if loss == 'labels':
                    # Logging is enabled only for the last layer
                    kwargs['log'] = False
                l_dict = self.get_loss(loss, enc_outputs, bin_targets, indices, indices_iou, num_boxes, **kwargs)
                l_dict = {k + f'_enc': v for k, v in l_dict.items()}
                losses.update(l_dict)

        return losses, indices

class InstanceLevelContrastiveLoss(nn.Module):
    r"""Contrastive loss for instance-level embeddings
    :param tau: temperature parameter for the similarity
    """
    def __init__(self, tau: float = 0.1):
        super(InstanceLevelContrastiveLoss, self).__init__()
        assert tau > 0, "tau should be positive"
        self.tau = tau

    def forward(self, embeddings: Tensor, labels: Tensor, iou_weight: Tensor):
        """Compute instance-level contrastive loss.
        
        Args:
            embeddings: Tensor of shape (num_embeddings, embed_dim)
            labels: Tensor of shape (num_embeddings,) containing instance labels
            iou_weight: Tensor of shape (num_embeddings,) containing IoU weights for each embedding
        
        Returns:
            loss_value: Scalar tensor representing the contrastive loss
        """
        # Ensure all tensors are on the same device
        device = embeddings.device
        labels = labels.to(device)
        iou_weight = iou_weight.to(device)
        assert embeddings.shape[0] == labels.shape[0], "Each embedding should have a label"
        num_embeddings = embeddings.shape[0]

        # positive pairs are the pairs of embeddings that have the same label
        positive_pairs = torch.eq(labels[None, :], labels[:, None])  # (num_embeddings, num_embeddings)
        positive_pairs[torch.arange(num_embeddings, device=device), torch.arange(num_embeddings, device=device)] = False
        pos_indices = positive_pairs.nonzero()  # get the indices of the positive pairs [[i, j], [i, k], ...]
        num_positives = positive_pairs.sum(dim=1)  # (num_embeddings, )
        if num_positives.sum() == 0:
            warnings.warn("No positive object pairs within the minibatch")
            
        weight_i = iou_weight[pos_indices[:, 0]]  # IoU weight of the first embedding
        weight_j = iou_weight[pos_indices[:, 1]]  # IoU weight of the second embedding
        pair_weights = 2 * weight_i * weight_j / (weight_i + weight_j)  # Harmonic mean of IoU weights

        # compute the similarity matrix
        embeddings = F.normalize(embeddings, dim=1)
        similarities = embeddings @ embeddings.t()
        similarities = (similarities / self.tau)

        # set the similarity of the diagonal to -inf
        similarities[torch.arange(num_embeddings, device=device), torch.arange(num_embeddings, device=device)] = float('-inf')

        # clone similarities and repeat num_positives times along the batch dimension
        neg_similarities = similarities.clone().repeat_interleave(num_positives, dim=0)
        neg_mask = positive_pairs.clone().repeat_interleave(num_positives, dim=0)
        neg_mask[torch.arange(neg_similarities.shape[0], device=device), pos_indices[:, 1]] = False
        neg_similarities[neg_mask] = float('-inf')

        # compute loss using "log sum exp" formulation for stability
        logsumexp = torch.logsumexp(neg_similarities, dim=1, keepdim=False)
        logprob = - (similarities[pos_indices[:, 0], pos_indices[:, 1]] - logsumexp)
        # calculate the weighted log probability
        weighted_logprob = logprob * pair_weights

        # average the loss over the positive pairs
        # TODO
        loss = weighted_logprob / num_positives[pos_indices[:, 0]]

        # average the loss over the batch elements with positive pairs
        loss_value = self.tau * loss.sum() / torch.clamp((num_positives != 0).sum(), min=1.)
        

        return loss_value

class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = out_logits.sigmoid()
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), 100, dim=1)
        scores = topk_values
        topk_boxes = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))

        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

        return results


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(args):
    # num_classes = 20 if args.dataset_file != 'coco' else 91
    # if args.dataset_file == "coco_panoptic":
    #     num_classes = 250
    num_classes = args.num_classes
    device = torch.device(args.device)

    backbone = build_backbone(args)
    depth_predictor = DepthPredictor(args)

    transformer = build_deforamble_transformer(args)
    model = DeformableDETR(
        backbone,
        depth_predictor,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        num_feature_levels=args.num_feature_levels,
        aux_loss=args.aux_loss,
        with_box_refine=args.with_box_refine,
        two_stage=args.two_stage,
    )
    if args.masks:
        model = DETRsegm(model, freeze_detr=(args.frozen_weights is not None))
    matcher = build_matcher(args)
    weight_dict = {'loss_ce': args.cls_loss_coef, 'loss_bbox': args.bbox_loss_coef}
    weight_dict['loss_giou'] = args.giou_loss_coef
    weight_dict['loss_depth_map'] = args.depth_loss_coef
    weight_dict['loss_contrastive'] = args.contrastive_loss_coef
    if args.masks:
        weight_dict["loss_mask"] = args.mask_loss_coef
        weight_dict["loss_dice"] = args.dice_loss_coef
    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        aux_weight_dict.update({k + f'_enc': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes', 'cardinality', 'depth_map', 'contrastive']
    if args.masks:
        losses += ["masks"]
    # num_classes, matcher, weight_dict, losses, focal_alpha=0.25
    criterion = SetCriterion(num_classes, matcher, weight_dict, losses, 
                             ddn_mode=args.depth_mode, fg_weight=args.ddn_fg_weight,
                             num_bins=args.depth_bin_num, focal_alpha=args.focal_alpha)
    criterion.to(device)
    postprocessors = {'bbox': PostProcess()}
    if args.masks:
        postprocessors['segm'] = PostProcessSegm()
        if args.dataset_file == "coco_panoptic":
            is_thing_map = {i: i <= 90 for i in range(201)}
            postprocessors["panoptic"] = PostProcessPanoptic(is_thing_map, threshold=0.85)

    return model, criterion, postprocessors

