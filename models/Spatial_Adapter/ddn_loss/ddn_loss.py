import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import cv2

from .balancer import Balancer
from .focalloss import FocalLoss

# based on:
# https://github.com/TRAILab/CaDDN/blob/master/pcdet/models/backbones_3d/ffe/ddn_loss/ddn_loss.py


class DDNLoss(nn.Module):

    def __init__(self,
                 alpha=0.25,
                 gamma=2.0,
                 fg_weight=6,
                 bg_weight=1,
                 ddn_mode="LID",
                 num_bins=150,
                 downsample_factor=1):
        """
        Initializes DDNLoss module
        Args:
            weight [float]: Loss function weight
            alpha [float]: Alpha value for Focal Loss
            gamma [float]: Gamma value for Focal Loss
            disc_cfg [dict]: Depth discretiziation configuration
            fg_weight [float]: Foreground loss weight
            bg_weight [float]: Background loss weight
            ddn_mode [str]: Depth discretization mode
            num_bins [int]: Number of depth bins
            downsample_factor [int]: Depth map downsample factor
        """
        super().__init__()
        self.device = torch.cuda.current_device()
        self.balancer = Balancer(
            downsample_factor=downsample_factor,
            fg_weight=fg_weight,
            bg_weight=bg_weight)

        # Set loss function
        self.alpha = alpha
        self.gamma = gamma
        self.ddn_mode = ddn_mode
        self.num_bins = num_bins
        self.loss_func = FocalLoss(alpha=self.alpha, gamma=self.gamma, reduction="none")

    def reshape_depth_maps(self, depth_logits, depthmaps):
        """
        Load depth maps from files for the current batch and resize to match depth_logits
        Args:
            depth_logits: Predicted depth logits from the network
            depth_maps: Information about current batch (frame IDs, sequence names, etc.)
        Returns:
            results List[torch.Tensor(1, H, W)]: Loaded depth maps resized to match depth_logits
        """
        _, _, H, W = depth_logits.shape  # Get dimensions from depth_logits directly
        results = []
        
        for depth_map in depthmaps:
            # Resize to match depth_logits dimensions
            depth_map = F.interpolate(depth_map[None], (H, W), mode='bilinear', align_corners=False)
            results.append(depth_map[0])
        return torch.cat(results, dim=0)

    def bin_depths(self, depth_map, mode="LID", depth_min=1e-3, depth_max=256.0, num_bins=150, target=False):
        """
        Converts depth map into bin indices
        Args:
            depth_map [torch.Tensor(H, W)]: Depth Map
            mode [string]: Discretiziation mode (See https://arxiv.org/pdf/2005.13423.pdf for more details)
                UD: Uniform discretiziation
                LID: Linear increasing discretiziation
                SID: Spacing increasing discretiziation
            depth_min [float]: Minimum depth value
            depth_max [float]: Maximum depth value
            num_bins [int]: Number of depth bins
            target [bool]: Whether the depth bins indices will be used for a target tensor in loss comparison
        Returns:
            indices [torch.Tensor(H, W)]: Depth bin indices
        """
        if mode == "UD":
            bin_size = (depth_max - depth_min) / num_bins
            indices = ((depth_map - depth_min) / bin_size)
        elif mode == "LID":
            bin_size = 2 * (depth_max - depth_min) / (num_bins * (1 + num_bins))
            indices = -0.5 + 0.5 * torch.sqrt(1 + 8 * (depth_map - depth_min) / bin_size)
        elif mode == "SID":
            indices = num_bins * (torch.log(1 + depth_map) - math.log(1 + depth_min)) / \
                      (math.log(1 + depth_max) - math.log(1 + depth_min))
        else:
            raise NotImplementedError

        if target:
            # Remove indicies outside of bounds
            mask = (indices < 0) | (indices > num_bins) | (~torch.isfinite(indices))
            indices[mask] = num_bins

            # Convert to integer
            indices = indices.type(torch.int64)
       
        return indices

    def forward(self, depth_logits, gt_boxes2d, num_gt_per_img, depthmaps):
        """
        Gets depth_map loss
        Args:
            depth_logits: torch.Tensor(B, D+1, H, W)]: Predicted depth logits
            gt_boxes2d [torch.Tensor (B, N, 4)]: 2D box labels for foreground/background balancing
            num_gt_per_img:
            gt_center_depth:
        Returns:
            loss [torch.Tensor(1)]: Depth classification network loss
        """

        # Bin depth map to create target
        depth_maps = self.reshape_depth_maps(depth_logits, depthmaps)
        depth_target = self.bin_depths(depth_maps, mode=self.ddn_mode, 
                                       num_bins=self.num_bins, target=True)
        # Compute loss
        loss = self.loss_func(depth_logits, depth_target)
        # Compute foreground/background balancing
        loss = self.balancer(loss=loss, gt_boxes2d=gt_boxes2d, num_gt_per_img=num_gt_per_img)

        return loss
