"""
Utility Functions (Config, Bounding Boxes, WBF, EMA, Checkpointing, and Visualizations)
Author: Antigravity AI Pair Programmer
"""

import os
import yaml
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

class EMA:
    """
    Exponential Moving Average (EMA) of model parameters.
    Keeps a smoothed copy of model weights for validation and inference.
    """
    def __init__(self, model, decay: float = 0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()

    def register(self):
        """Register copy of parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        """Update shadow parameters with current parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        """Apply shadow weights to model for evaluation."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self):
        """Restore original parameters after evaluation."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data.copy_(self.backup[name])
        self.backup = {}

def xywh2xyxy(x):
    """Convert bounding boxes from [x_center, y_center, w, h] to [x_min, y_min, x_max, y_max]"""
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # x min
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # y min
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # x max
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # y max
    return y

def box_iou(box1, box2):
    """Compute IoU between two box groups in [x_min, y_min, x_max, y_max] format."""
    # Intersection dimensions
    lt = np.maximum(box1[:, None, :2], box2[:, :2])
    rb = np.minimum(box1[:, None, 2:], box2[:, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[:, :, 0] * wh[:, :, 1]
    
    # Area
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    union = area1[:, None] + area2 - inter + 1e-16
    
    return inter / union

def non_maximum_suppression(prediction, conf_thres=0.25, iou_thres=0.45, max_det=300):
    """
    Standard Non-Maximum Suppression (NMS) over multi-scale model predictions.
    prediction: List of scale heads decoded results, e.g., predictions shape [B, A, H, W, 5+C]
    Returns list of filtered boxes per image: [x_min, y_min, x_max, y_max, confidence, class_id]
    """
    # Flaten predictions into [B, A*H*W, 5+C]
    flat_preds = []
    for pred in prediction:
        B, A, H, W, _ = pred.shape
        flat = pred.view(B, A * H * W, -1)
        flat_preds.append(flat)
    merged_pred = torch.cat(flat_preds, dim=1) # [B, Total_Boxes, 5+C]
    
    device = merged_pred.device
    num_classes = merged_pred.shape[-1] - 5
    output = [torch.zeros((0, 6), device=device) for _ in range(merged_pred.shape[0])]
    
    for image_idx, img_pred in enumerate(merged_pred):
        # Extract objectness confidence score
        obj_conf = torch.sigmoid(img_pred[:, 4])
        # Filter by confidence threshold
        conf_mask = obj_conf > conf_thres
        img_pred = img_pred[conf_mask]
        if img_pred.size(0) == 0:
            continue
            
        # Coordinates in [x_center, y_center, w, h] -> convert to xyxy
        boxes_xyxy = xywh2xyxy(img_pred[:, :4])
        
        # Class scores: sigmoid(class_logits) * objectness
        class_probs = torch.sigmoid(img_pred[:, 5:]) * obj_conf.unsqueeze(1)[conf_mask]
        
        # Get highest probability class per box
        max_prob, class_ids = torch.max(class_probs, dim=1)
        
        # Retain elements above threshold
        prob_mask = max_prob > conf_thres
        boxes = boxes_xyxy[prob_mask]
        scores = max_prob[prob_mask]
        labels = class_ids[prob_mask]
        
        if boxes.size(0) == 0:
            continue
            
        # standard NMS using torchvision
        from torchvision.ops import nms
        # Batched NMS to perform NMS separately for each class
        # Add offset relative to class_id to prevent suppression across classes
        offsets = labels.to(device, dtype=torch.float32) * 4096.0
        nms_boxes = boxes + offsets.unsqueeze(1)
        
        keep = nms(nms_boxes, scores, iou_thres)
        keep = keep[:max_det]
        
        final_boxes = boxes[keep]
        final_scores = scores[keep]
        final_labels = labels[keep]
        
        output[image_idx] = torch.cat([
            final_boxes, 
            final_scores.unsqueeze(1), 
            final_labels.unsqueeze(1).to(torch.float32)
        ], dim=1)
        
    return output

# 14 Clinical colors for radiologist visual support (Vibrant RGB palette)
PATHOLOGY_COLORS = [
    (244, 67, 54),    # Aortic enlargement: Bright Red
    (233, 30, 99),    # Atelectasis: Deep Pink
    (156, 39, 176),   # Calcification: Purple
    (103, 58, 183),   # Cardiomegaly: Deep Indigo
    (63, 81, 181),    # Consolidation: Indigo
    (33, 150, 243),   # ILD: Blue
    (0, 188, 212),    # Infiltration: Cyan
    (0, 150, 136),    # Lung Opacity: Teal
    (76, 175, 80),    # Nodule/Mass: Bright Green
    (139, 195, 74),   # Other lesion: Lime Green
    (255, 235, 59),   # Pleural effusion: Vivid Yellow
    (255, 152, 0),    # Pleural thickening: Amber/Orange
    (255, 87, 34),    # Pneumothorax: Deep Orange
    (121, 85, 72)     # Pulmonary fibrosis: Soft Brown
]

def draw_predictions(image: np.ndarray, detections: np.ndarray, classes: list) -> np.ndarray:
    """
    Draw 14-color bounding box representations on chest radiographies.
    image: np.ndarray of shape [H, W, 3] (RGB)
    detections: np.ndarray of shape [N, 6] -> columns [x_min, y_min, x_max, y_max, conf, class_id]
    classes: list of 14 class names
    """
    canvas = image.copy()
    H, W, _ = canvas.shape
    
    for det in detections:
        x_min, y_min, x_max, y_max, conf, class_id = det
        class_id = int(class_id)
        
        # Bound coords to image scope
        x_min = max(0, int(x_min))
        y_min = max(0, int(y_min))
        x_max = min(W, int(x_max))
        y_max = min(H, int(y_max))
        
        label_name = classes[class_id]
        color = PATHOLOGY_COLORS[class_id % len(PATHOLOGY_COLORS)]
        
        # Bounding box
        cv2.rectangle(canvas, (x_min, y_min), (x_max, y_max), color, thickness=3)
        
        # Text label with backdrop
        text = f"{label_name} {conf:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        # Draw text background
        cv2.rectangle(
            canvas, 
            (x_min, max(0, y_min - text_h - 10)), 
            (x_min + text_w + 10, y_min), 
            color, 
            cv2.FILLED
        )
        # Draw white text
        cv2.putText(
            canvas, 
            text, 
            (x_min + 5, max(12, y_min - 5)), 
            font, 
            font_scale, 
            (255, 255, 255), 
            thickness, 
            lineType=cv2.LINE_AA
        )
        
    return canvas

def save_checkpoint(state, is_best, checkpoint_dir, filename="checkpoint.pth"):
    """Save full training state and keep separate best_model file."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)
    if is_best:
        best_path = os.path.join(checkpoint_dir, "best_model.pth")
        torch.save(state, best_path)
        print(f"⭐️ Saved new best checkpoint to {best_path}")

def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None, ema=None):
    """Load model weight dictionaries and optimization states from checkpoint."""
    print(f"📂 Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    if ema is not None and 'ema_state_dict' in checkpoint:
        ema.shadow = checkpoint['ema_state_dict']
        
    start_epoch = checkpoint.get('epoch', 0)
    best_loss = checkpoint.get('best_val_loss', float('inf'))
    
    print(f"✅ Loaded checkpoint successfully (Started at epoch {start_epoch}, best loss {best_loss:.4f})")
    return start_epoch, best_loss
