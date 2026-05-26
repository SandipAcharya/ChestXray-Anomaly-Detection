"""
PyTorch Dataset & Albumentations Augmentations Pipeline
Author: Antigravity AI Pair Programmer
"""

import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class ChestXrayDataset(Dataset):
    """
    Custom PyTorch Dataset for loading multi-view chest X-rays,
    applying medical augmentations, and formatting targets for CMRF-Net.
    """
    def __init__(
        self,
        csv_path: str,
        image_dir: str,
        input_size: int = 512,
        is_training: bool = True,
        classes: list = None
    ):
        super().__init__()
        self.csv_path = csv_path
        self.image_dir = image_dir
        self.input_size = input_size
        self.is_training = is_training
        
        # Load standard class list from config
        self.classes = classes if classes is not None else [
            "Aortic enlargement", "Atelectasis", "Calcification", "Cardiomegaly",
            "Consolidation", "ILD", "Infiltration", "Lung Opacity", "Nodule/Mass",
            "Other lesion", "Pleural effusion", "Pleural thickening", "Pneumothorax",
            "Pulmonary fibrosis"
        ]
        self.class_to_id = {c: i for i, c in enumerate(self.classes)}
        
        # Load CSV annotations
        print(f"Reading dataset annotations from {csv_path}...")
        self.df = pd.read_csv(csv_path)
        
        # Group bounding boxes by unique image_id
        self.image_ids = self.df["image_id"].unique().tolist()
        self._group_annotations()
        
        # Build Albumentations Augmentation pipeline
        self.transform = self._get_transforms()

    def _group_annotations(self):
        """Pre-group annotations by image_id for high-speed indexing during training."""
        self.annotations = {}
        for img_id, group in self.df.groupby("image_id"):
            boxes = []
            classes_present = set()
            
            for _, row in group.iterrows():
                class_name = row["class_name"]
                
                # Check for "No Finding"
                if class_name.lower() == "no finding" or pd.isna(row["x_min"]):
                    continue
                    
                # Bounding boxes are stored as [x_min, y_min, x_max, y_max] in pixels
                x_min = float(row["x_min"])
                y_min = float(row["y_min"])
                x_max = float(row["x_max"])
                y_max = float(row["y_max"])
                
                # Avoid inverted or invalid coordinates
                if x_max <= x_min or y_max <= y_min:
                    continue
                    
                class_id = self.class_to_id[class_name]
                boxes.append([x_min, y_min, x_max, y_max, class_id])
                classes_present.add(class_id)
                
            self.annotations[img_id] = {
                "boxes": np.array(boxes, dtype=np.float32) if boxes else np.empty((0, 5), dtype=np.float32),
                "classes_present": list(classes_present)
            }

    def _get_transforms(self):
        """Define albumentations augmentations tailored for grayscale chest radiographies."""
        if self.is_training:
            return A.Compose([
                # Spatial Transformations
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.1, 
                    scale_limit=0.15, 
                    rotate_limit=15, 
                    border_mode=cv2.BORDER_CONSTANT, 
                    value=0, 
                    p=0.5
                ),
                
                # Intensity transformations ( Exposure / gamma / noise simulations )
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.RandomGamma(gamma_limit=(80, 120), p=0.3),
                A.GaussianBlur(p=0.2),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
                
                # Local contrast adaptations
                A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
                A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.2),
                
                # Resize and normalization
                A.Resize(self.input_size, self.input_size),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='pascal_voc', min_area=1.0, min_visibility=0.3, label_fields=['class_labels']))
        else:
            return A.Compose([
                A.Resize(self.input_size, self.input_size),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='pascal_voc', min_area=1.0, min_visibility=0.3, label_fields=['class_labels']))

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        
        # Load image (Grayscale or converted from RGB)
        image_path = os.path.join(self.image_dir, f"{image_id}.png")
        if not os.path.exists(image_path):
            # Fallback for alternative extensions
            for ext in [".jpg", ".jpeg", ".bmp"]:
                fallback_path = os.path.join(self.image_dir, f"{image_id}{ext}")
                if os.path.exists(fallback_path):
                    image_path = fallback_path
                    break
                    
        image = cv2.imread(image_path)
        if image is None:
            # Return dummy elements if loading fails to prevent crash
            image = np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
        # Get annotations
        ann = self.annotations.get(image_id, {"boxes": np.empty((0, 5)), "classes_present": []})
        gt_boxes = ann["boxes"]
        classes_present = ann["classes_present"]
        
        # Prepare targets
        bboxes = []
        class_labels = []
        if len(gt_boxes) > 0:
            bboxes = gt_boxes[:, :4].tolist()
            class_labels = gt_boxes[:, 4].astype(int).tolist()
            
        # Apply transforms
        try:
            transformed = self.transform(image=image, bboxes=bboxes, class_labels=class_labels)
            img_tensor = transformed["image"]
            transformed_boxes = transformed["bboxes"]
            transformed_labels = transformed["class_labels"]
        except Exception:
            # Fallback in case of boundary violations
            transformed = self.transform(image=image, bboxes=[], class_labels=[])
            img_tensor = transformed["image"]
            transformed_boxes = []
            transformed_labels = []
            
        # Re-format bounding boxes for YOLO-style targets: [class_id, x_center, y_center, width, height] normalized
        targets = []
        for box, label in zip(transformed_boxes, transformed_labels):
            x_min, y_min, x_max, y_max = box
            
            # Convert to normalized xywh format relative to input size
            x_center = (x_min + x_max) / 2.0 / self.input_size
            y_center = (y_min + y_max) / 2.0 / self.input_size
            width = (x_max - x_min) / self.input_size
            height = (y_max - y_min) / self.input_size
            
            targets.append([label, x_center, y_center, width, height])
            
        if len(targets) > 0:
            targets_tensor = torch.tensor(targets, dtype=torch.float32)
        else:
            targets_tensor = torch.zeros((0, 5), dtype=torch.float32)
            
        # Construct Global Triage multi-label vector (15 classes: 14 pathologies + 1 "No Finding")
        triage_target = torch.zeros(15, dtype=torch.float32)
        if len(classes_present) > 0:
            for c_id in classes_present:
                triage_target[c_id] = 1.0
        else:
            # If no regional pathologies exist, trigger the "No Finding" class (Index 14)
            triage_target[14] = 1.0
            
        return img_tensor, targets_tensor, triage_target

def collate_fn(batch):
    """
    Custom collate function to handle variable number of bounding boxes per image.
    Packs variable-length target bounding boxes with a batch index.
    """
    images, targets, triage_targets = zip(*batch)
    
    # Collate images and triage vectors
    collated_images = torch.stack(images, dim=0)
    collated_triage = torch.stack(triage_targets, dim=0)
    
    # Collate bounding box targets. Add a batch index prefix to each bounding box.
    # Yields: [N_total_boxes_in_batch, 6] where columns are [batch_idx, class_id, x_center, y_center, w, h]
    collated_targets = []
    for i, tgt in enumerate(targets):
        if tgt.size(0) > 0:
            batch_idx = torch.full((tgt.size(0), 1), i, dtype=torch.float32)
            # tgt format is [class_id, x_center, y_center, w, h]
            tgt_with_idx = torch.cat([batch_idx, tgt], dim=1)
            collated_targets.append(tgt_with_idx)
            
    if len(collated_targets) > 0:
        collated_targets = torch.cat(collated_targets, dim=0)
    else:
        collated_targets = torch.zeros((0, 6), dtype=torch.float32)
        
    return collated_images, collated_targets, collated_triage
