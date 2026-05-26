"""
Custom Multi-Scale Dynamic CIoU & Focal Loss Module
Author: Antigravity AI Pair Programmer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CMRFLoss(nn.Module):
    """
    Vectorized Custom Loss for CMRF-Net.
    Calculates CIoU bounding box regression, Focal Loss on objectness (with an ignore threshold),
    multi-label class BCE, and global auxiliary triage loss.
    """
    def __init__(self, num_classes: int = 14, triage_classes: int = 15, config: dict = None):
        super().__init__()
        self.num_classes = num_classes
        self.triage_classes = triage_classes
        
        # Load weights from config — verified against doc Table 3
        self.lambda_box    = config.get("lambda_box",    5.0) if config else 5.0
        self.lambda_obj    = config.get("lambda_obj",    1.0) if config else 1.0
        self.lambda_noobj  = config.get("lambda_noobj",  0.5) if config else 0.5  # doc Table 3
        self.lambda_cls    = config.get("lambda_cls",    1.0) if config else 1.0  # doc Table 3: 1.0 not 0.5
        self.lambda_triage = config.get("lambda_triage", 1.0) if config else 1.0

        self.focal_gamma   = config.get("focal_gamma",   2.0) if config else 2.0
        self.focal_alpha   = config.get("focal_alpha",   0.25) if config else 0.25
        self.ignore_iou_thr = config.get("ignore_iou_thr", 0.5) if config else 0.5
        
        # Anchor Box Configurations (normalized to [0, 1] relative to 512x512)
        # Small anchors (Stage 7 - 256x256 grid)
        # Medium anchors (Stage 8 - 128x128 grid)
        # Large anchors (Stage 9 - 64x64 grid)
        self.anchors = [
            [[10/512, 15/512], [20/512, 36/512], [41/512, 31/512]],       # Head 0 (High-res / grid 256)
            [[77/512, 56/512], [51/512, 108/512], [113/512, 174/512]],    # Head 1 (Med-res / grid 128)
            [[246/512, 317/512], [369/512, 215/512], [451/512, 481/512]]  # Head 2 (Low-res / grid 64)
        ]
        
        # Flattend list of all 9 anchors for scaling/matching
        self.all_anchors = torch.tensor(
            self.anchors[0] + self.anchors[1] + self.anchors[2], 
            dtype=torch.float32
        )

    def _bbox_iou(self, box1, box2, x1y1x2y2=True, CIoU=True):
        """
        Calculate CIoU/IoU between two bounding boxes.
        box1: Predicted bounding box [N, 4]
        box2: Target bounding box [N, 4]
        """
        if x1y1x2y2:
            b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
            b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]
        else:
            # xywh format
            b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
            b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
            b2_x1, b2_y1 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
            b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2

        # Intersection area
        inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
                (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

        # Union Area
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1
        union = w1 * h1 + w2 * h2 - inter + 1e-16

        iou = inter / union
        if not CIoU:
            return iou

        # Smallest enclosing box
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)
        c2 = cw ** 2 + ch ** 2 + 1e-16  # convex diagonal squared

        # Center distance squared
        rho2 = ((b1_x1 + b1_x2 - b2_x1 - b2_x2) ** 2 + (b1_y1 + b1_y2 - b2_y1 - b2_y2) ** 2) / 4

        # Aspect ratio penalty
        v = (4 / (math.pi ** 2)) * torch.pow(torch.atan(w2 / (h2 + 1e-16)) - torch.atan(w1 / (h1 + 1e-16)), 2)
        with torch.no_grad():
            alpha = v / ((1 - iou) + v + 1e-16)

        return iou - (rho2 / c2) - alpha * v

    def _decode_predictions(self, pred, anchors_scale, grid_size):
        """
        Decodes model predictions [B, A, H, W, 5+C] into normalized xywh format.
        """
        device = pred.device
        B, A, H, W, _ = pred.shape
        
        # Grid cell offset coordinate grids
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=device), 
            torch.arange(W, device=device), 
            indexing='ij'
        )
        grid_x = grid_x.view(1, 1, H, W, 1).repeat(B, A, 1, 1, 1)
        grid_y = grid_y.view(1, 1, H, W, 1).repeat(B, A, 1, 1, 1)
        
        # Rescale anchors to grid cell dimensions
        anchors_tensor = torch.tensor(anchors_scale, device=device, dtype=torch.float32) # [A, 2]
        anchors_w = anchors_tensor[:, 0].view(1, A, 1, 1, 1).repeat(B, 1, H, W, 1)
        anchors_h = anchors_tensor[:, 1].view(1, A, 1, 1, 1).repeat(B, 1, H, W, 1)
        
        # Extract components
        pred_xy = torch.sigmoid(pred[..., 0:2])  # Center offsets relative to grid cells
        pred_wh = pred[..., 2:4]                 # Logarithmic width/height offsets
        
        # Coordinate conversions
        dec_x = (pred_xy[..., 0:1] + grid_x) / W
        dec_y = (pred_xy[..., 1:2] + grid_y) / H
        dec_w = torch.exp(pred_wh[..., 0:1]) * anchors_w
        dec_h = torch.exp(pred_wh[..., 1:2]) * anchors_h
        
        decoded_boxes = torch.cat([dec_x, dec_y, dec_w, dec_h], dim=-1)
        return decoded_boxes

    def _anchor_matching(self, targets, device):
        """
        Compute optimal anchor assignments dynamically for ground truth targets.
        targets: [N_total_boxes, 6] where columns are [batch_idx, class_id, x_center, y_center, w, h]
        Returns matching indices list for [head_0, head_1, head_2]
        """
        num_targets = targets.size(0)
        matching_targets = [[] for _ in range(3)] # Heads 0, 1, 2
        
        if num_targets == 0:
            return matching_targets
            
        self.all_anchors = self.all_anchors.to(device)
        
        # Extract target dimensions
        target_wh = targets[:, 4:6] # [N, 2]
        
        # Calculate overlap (IoU) of target boxes with all 9 anchors at the origin
        # Intersection width/height
        inter_w = torch.min(target_wh[:, 0].unsqueeze(1), self.all_anchors[:, 0].unsqueeze(0))
        inter_h = torch.min(target_wh[:, 1].unsqueeze(1), self.all_anchors[:, 1].unsqueeze(0))
        intersection = inter_w * inter_h
        
        # Union area
        target_area = target_wh[:, 0] * target_wh[:, 1]
        anchor_area = self.all_anchors[:, 0] * self.all_anchors[:, 1]
        union = target_area.unsqueeze(1) + anchor_area.unsqueeze(0) - intersection
        
        # IoU matrix [N_targets, 9_anchors]
        ious = intersection / (union + 1e-16)
        
        # Designate matching anchor (the index in 0-8 with max IoU)
        best_anchor_idx = torch.argmax(ious, dim=1) # [N]
        
        for idx in range(num_targets):
            best_a = best_anchor_idx[idx].item()
            head_idx = best_a // 3          # Head 0, 1, or 2
            anchor_in_head = best_a % 3     # Local anchor idx 0, 1, or 2
            
            tgt = targets[idx]
            batch_idx = int(tgt[0].item())
            class_id = int(tgt[1].item())
            x, y, w, h = tgt[2:6]
            
            matching_targets[head_idx].append({
                'batch_idx': batch_idx,
                'class_id': class_id,
                'x': x.item(), 'y': y.item(), 'w': w.item(), 'h': h.item(),
                'anchor_idx': anchor_in_head
            })
            
        return matching_targets

    def forward(self, predictions, targets, triage_targets):
        """
        Forward loss calculations.
        predictions: (pred_high, pred_med, pred_low, global_triage)
        targets: Packed ground truth bounding boxes [N_total_boxes, 6]
        triage_targets: Ground truth triage vectors [B, 15]
        """
        pred_high, pred_med, pred_low, global_triage = predictions
        device = pred_high.device
        batch_size = pred_high.size(0)
        
        # Compile predictions list
        preds = [pred_high, pred_med, pred_low]
        grid_sizes = [256, 128, 64]
        
        # Calculate dynamic anchor pairings
        matching_targets = self._anchor_matching(targets, device)
        
        total_loss_box = torch.tensor(0.0, device=device)
        total_loss_obj = torch.tensor(0.0, device=device)
        total_loss_cls = torch.tensor(0.0, device=device)
        
        for head_idx, (pred, grid_size, anchor_scale) in enumerate(zip(preds, grid_sizes, self.anchors)):
            B, A, H, W, _ = pred.shape
            
            # Decode model predictions
            decoded_boxes = self._decode_predictions(pred, anchor_scale, grid_size) # [B, A, H, W, 4]
            
            # Extract objectness logits and class logits
            obj_logits = pred[..., 4]              # [B, A, H, W]
            cls_logits = pred[..., 5:]             # [B, A, H, W, C]
            
            # Initialize target tensors
            target_obj = torch.zeros((B, A, H, W), device=device, dtype=torch.float32)
            target_cls = torch.zeros((B, A, H, W, self.num_classes), device=device, dtype=torch.float32)
            weight_mask = torch.ones((B, A, H, W), device=device, dtype=torch.float32)
            
            head_matches = matching_targets[head_idx]
            
            # 1. Fill positive targets
            pos_pred_boxes = []
            pos_target_boxes = []
            pos_cls_logits = []
            pos_cls_targets = []
            
            for match in head_matches:
                b = match['batch_idx']
                c = match['class_id']
                a = match['anchor_idx']
                gx = min(int(match['x'] * W), W - 1)
                gy = min(int(match['y'] * H), H - 1)
                
                # Set positive objectness target
                target_obj[b, a, gy, gx] = 1.0
                
                # Set positive class targets
                target_cls[b, a, gy, gx, c] = 1.0
                
                # Bounding box coordinates for CIoU
                p_box = decoded_boxes[b, a, gy, gx].unsqueeze(0) # [1, 4] (xywh)
                t_box = torch.tensor([match['x'], match['y'], match['w'], match['h']], device=device, dtype=torch.float32).unsqueeze(0)
                
                pos_pred_boxes.append(p_box)
                pos_target_boxes.append(t_box)
                
                # Class predictions for BCE
                pos_cls_logits.append(cls_logits[b, a, gy, gx].unsqueeze(0))
                pos_cls_targets.append(target_cls[b, a, gy, gx].unsqueeze(0))
                
            # 2. Implement the Ignore Threshold Strategy
            # Mask out background grids containing high overlaps that are not the matched prediction
            # Iterate per image to handle independent bounding box scopes
            for b in range(batch_size):
                img_targets = targets[targets[:, 0] == b]
                if img_targets.size(0) == 0:
                    continue
                    
                # [N_img_targets, 4]
                img_tgt_xywh = img_targets[:, 2:6]
                
                # Flatten head predictions for this image: [3 * H * W, 4]
                flat_pred = decoded_boxes[b].view(-1, 4)
                
                # Compute IoU between all predictions and all ground truths in the image
                # Resulting overlap matrix shape: [3*H*W, N_img_targets]
                flat_pred_expanded = flat_pred.unsqueeze(1).repeat(1, img_tgt_xywh.size(0), 1)
                img_tgt_expanded = img_tgt_xywh.unsqueeze(0).repeat(flat_pred.size(0), 1, 1)
                
                # Calculate IoUs
                ious = self._bbox_iou(
                    flat_pred_expanded.view(-1, 4), 
                    img_tgt_expanded.view(-1, 4), 
                    x1y1x2y2=False, 
                    CIoU=False
                ).view(flat_pred.size(0), img_tgt_xywh.size(0))
                
                # Find maximum IoU overlap per grid cell
                max_ious, _ = torch.max(ious, dim=1) # [3 * H * W]
                max_ious = max_ious.view(A, H, W)
                
                # Identify grids where overlap exceeds the ignore threshold but are NOT positive targets
                ignore_mask = (max_ious > self.ignore_iou_thr) & (target_obj[b] < 0.5)
                weight_mask[b][ignore_mask] = 0.0 # Suppress gradient loss
                
            # 3. Calculate Localization Loss (CIoU)
            if len(pos_pred_boxes) > 0:
                pos_pred_boxes = torch.cat(pos_pred_boxes, dim=0)
                pos_target_boxes = torch.cat(pos_target_boxes, dim=0)
                
                ciou = self._bbox_iou(pos_pred_boxes, pos_target_boxes, x1y1x2y2=False, CIoU=True)
                loss_box = torch.mean(1.0 - ciou)
                total_loss_box += loss_box
                
            # 4. Calculate Class BCE Loss
            if len(pos_cls_logits) > 0:
                pos_cls_logits = torch.cat(pos_cls_logits, dim=0)
                pos_cls_targets = torch.cat(pos_cls_targets, dim=0)
                
                loss_cls = F.binary_cross_entropy_with_logits(pos_cls_logits, pos_cls_targets, reduction='mean')
                total_loss_cls += loss_cls
                
            # 5. Objectness Focal Loss — split into Lobj and Lnoobj (doc eq. line 175, Table 3)
            bce_all = F.binary_cross_entropy_with_logits(obj_logits, target_obj, reduction='none')
            pred_prob = torch.sigmoid(obj_logits)
            p_t = pred_prob * target_obj + (1.0 - pred_prob) * (1.0 - target_obj)
            focal_weight = torch.pow(1.0 - p_t, self.focal_gamma)
            alpha_weight = target_obj * self.focal_alpha + (1.0 - target_obj) * (1.0 - self.focal_alpha)
            focal_loss_all = focal_weight * alpha_weight * bce_all

            # Positive (obj) cells
            obj_mask   = (target_obj > 0.5)
            # Negative (noobj) cells = not positive AND not ignored
            noobj_mask = (~obj_mask) & (weight_mask > 0.5)

            loss_obj   = focal_loss_all[obj_mask].mean()   if obj_mask.any()   else focal_loss_all.new_zeros(1).squeeze()
            loss_noobj = focal_loss_all[noobj_mask].mean() if noobj_mask.any() else focal_loss_all.new_zeros(1).squeeze()

            total_loss_obj += self.lambda_obj   * loss_obj
            total_loss_obj += self.lambda_noobj * loss_noobj
            
        # 6. Global Triage Classification Loss
        # Binary Cross Entropy over multi-label clinical triage targets
        loss_triage = F.binary_cross_entropy(global_triage, triage_targets, reduction='mean')
        
        # Combine loss — obj/noobj weights already applied inside the loop
        total_loss = (self.lambda_box    * total_loss_box) + \
                      total_loss_obj                      + \
                     (self.lambda_cls    * total_loss_cls) + \
                     (self.lambda_triage * loss_triage)
                     
        return total_loss, {
            'loss': total_loss.item(),
            'box_loss': total_loss_box.item(),
            'obj_loss': total_loss_obj.item(),
            'cls_loss': total_loss_cls.item(),
            'triage_loss': loss_triage.item()
        }
