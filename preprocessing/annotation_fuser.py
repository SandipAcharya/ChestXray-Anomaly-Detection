"""
Annotation Fuser Module (Enclosing Box Removal & Weighted Boxes Fusion)
Author: Antigravity AI Pair Programmer
"""

import os
import csv
import argparse
import random
from collections import Counter, defaultdict
from tqdm import tqdm

try:
    from ensemble_boxes import weighted_boxes_fusion
    HAS_WBF = True
except ImportError:
    HAS_WBF = False

def is_enclosing(boxA, boxB):
    """
    Check if boxA fully encloses boxB (boxB is entirely inside boxA).
    Each box is in format [x_min, y_min, x_max, y_max]
    """
    Axmin, Aymin, Axmax, Aymax = boxA
    Bxmin, Bymin, Bxmax, Bymax = boxB
    return (Axmin <= Bxmin) and (Aymin <= Bymin) and (Axmax >= Bxmax) and (Aymax >= Bymax)

def box_area(box):
    """Calculate area of a box [x_min, y_min, x_max, y_max]"""
    x_min, y_min, x_max, y_max = box
    return max(0.0, x_max - x_min) * max(0.0, y_max - y_min)

def remove_enclosing_boxes(boxes_with_labels):
    """
    Remove redundant boxes that fully enclose other smaller boxes of the same class.
    boxes_with_labels: List of tuples (class_name, x_min, y_min, x_max, y_max)
    """
    if len(boxes_with_labels) <= 1:
        return boxes_with_labels

    # Sort by box area ascending so smaller boxes are evaluated first
    boxes_with_labels = sorted(boxes_with_labels, key=lambda x: box_area(x[1:]))
    kept = []

    for i, item in enumerate(boxes_with_labels):
        label, *box = item
        is_encloser = False
        for j, other_item in enumerate(boxes_with_labels):
            if i == j:
                continue
            other_label, *other_box = other_item
            if is_enclosing(box, other_box):
                is_encloser = True
                break
        if not is_encloser:
            kept.append(item)
            
    return kept

def fuse_annotations(csv_path, output_path, target_size=512, input_scale=640, iou_thr=0.5):
    """
    Load annotations from csv_path, perform enclosing-box removal, apply WBF, 
    rescale coordinates to target_size, and write the output.
    """
    if not HAS_WBF:
        raise ImportError("ensemble-boxes package is required. Please install it using 'pip install ensemble-boxes'.")

    print(f"📂 Loading annotations from {csv_path}...")
    
    # Read CSV rows and group by image_id
    image_groups = defaultdict(list)
    with open(csv_path, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_id = row['image_id']
            class_name = row['class_name']
            
            # Check for empty boxes (No Finding)
            if not row['x_min'] or not row['y_min'] or not row['x_max'] or not row['y_max']:
                image_groups[image_id].append((class_name, None, None, None, None))
                continue
                
            x_min = float(row['x_min'])
            y_min = float(row['y_min'])
            x_max = float(row['x_max'])
            y_max = float(row['y_max'])
            
            # Rescale coordinate values if current scale differs from target scale
            if input_scale != target_size:
                scale_factor = target_size / float(input_scale)
                x_min *= scale_factor
                y_min *= scale_factor
                x_max *= scale_factor
                y_max *= scale_factor
                
            image_groups[image_id].append((class_name, x_min, y_min, x_max, y_max))

    fused_rows = []
    print("⚡ Starting Enclosing Box Removal & Weighted Boxes Fusion (WBF)...")
    
    for image_id, annotations in tqdm(image_groups.items(), desc="Fusing Annotations"):
        # Separate "No finding" and active pathology annotations
        no_finding_anns = [ann for ann in annotations if ann[0].lower() == "no finding"]
        active_anns = [ann for ann in annotations if ann[0].lower() != "no finding"]
        
        # Keep "No finding" annotations directly
        for label, *box in no_finding_anns:
            fused_rows.append({
                'image_id': image_id,
                'class_name': label,
                'x_min': '',
                'y_min': '',
                'x_max': '',
                'y_max': ''
            })
            
        if not active_anns:
            continue
            
        # Group active annotations by class name
        class_groups = defaultdict(list)
        for item in active_anns:
            label, *box = item
            class_groups[label].append(item)
            
        # Step 1: Remove enclosing boxes within each class group
        clean_class_groups = {}
        for label, items in class_groups.items():
            clean_class_groups[label] = remove_enclosing_boxes(items)
            
        # Map class names to dynamic integer IDs for WBF processing
        unique_classes = sorted(list(clean_class_groups.keys()))
        class_to_id = {c: i for i, c in enumerate(unique_classes)}
        id_to_class = {i: c for c, i in class_to_id.items()}
        
        boxes_single = []
        boxes_list = []
        scores_list = []
        labels_list = []
        weights = []
        
        # Step 2: Organize single box and multiple boxes for WBF
        for label, items in clean_class_groups.items():
            if len(items) == 1:
                # Keep single boxes without WBF
                _, *box = items[0]
                boxes_single.append((label, box))
            else:
                # Normalize box coordinates to [0, 1] for the ensemble-boxes package
                bboxes = [box for _, *box in items]
                bboxes_norm = [[b[0] / target_size, b[1] / target_size, b[2] / target_size, b[3] / target_size] for b in bboxes]
                
                boxes_list.append(bboxes_norm)
                scores_list.append([1.0] * len(items))
                labels_list.append([class_to_id[label]] * len(items))
                weights.append(1)
                
        # Perform WBF if multi-box classes are present
        if boxes_list:
            boxes_wbf, _, labels_wbf = weighted_boxes_fusion(
                boxes_list, scores_list, labels_list, weights=weights,
                iou_thr=iou_thr, skip_box_thr=0.0001
            )
            # Re-scale back to pixel coordinate system
            boxes_wbf = boxes_wbf * target_size
            boxes_wbf = boxes_wbf.round(2).tolist()
            labels_wbf = [id_to_class[int(l)] for l in labels_wbf]
            
            # Combine fused boxes and single boxes
            for box, label in zip(boxes_wbf, labels_wbf):
                fused_rows.append({
                    'image_id': image_id,
                    'class_name': label,
                    'x_min': box[0],
                    'y_min': box[1],
                    'x_max': box[2],
                    'y_max': box[3]
                })
                
        # Append single boxes
        for label, box in boxes_single:
            fused_rows.append({
                'image_id': image_id,
                'class_name': label,
                'x_min': round(box[0], 2),
                'y_min': round(box[1], 2),
                'x_max': round(box[2], 2),
                'y_max': round(box[3], 2)
            })

    # Write output CSV
    fieldnames = ['image_id', 'class_name', 'x_min', 'y_min', 'x_max', 'y_max']
    with open(output_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(fused_rows)
        
    print(f"✅ Successfully fused annotations! Saved output to: {output_path}")
    print(f"Total fused boxes: {len(fused_rows)}")

def split_and_save(fused_csv_path, output_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    """
    Split the fused annotations by unique image_id to ensure no patient overlap between splits.
    """
    random.seed(seed)
    
    # Group rows by image_id
    image_rows = defaultdict(list)
    with open(fused_csv_path, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            image_rows[row['image_id']].append(row)
            
    image_ids = list(image_rows.keys())
    random.shuffle(image_ids)
    
    total = len(image_ids)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    
    splits = {
        'train': image_ids[:train_end],
        'val': image_ids[train_end:val_end],
        'test': image_ids[val_end:]
    }
    
    os.makedirs(output_dir, exist_ok=True)
    
    for split_name, ids in splits.items():
        split_rows = []
        for img_id in ids:
            split_rows.extend(image_rows[img_id])
            
        split_csv = os.path.join(output_dir, f"{split_name}_annotations.csv")
        with open(split_csv, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(split_rows)
            
        print(f"📁 Split '{split_name}': {len(ids)} images ({len(split_rows)} annotations) -> {split_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Annotation Fusing and Splitting Pipeline")
    parser.add_argument("--csv", type=str, required=True, help="Path to original annotations CSV")
    parser.add_argument("--output", type=str, required=True, help="Path to save fused annotations CSV")
    parser.add_argument("--split_dir", type=str, default=None, help="Directory to save train/val/test CSV splits")
    parser.add_argument("--target_size", type=int, default=512, help="Target image size (default: 512)")
    parser.add_argument("--input_scale", type=int, default=640, help="Scale of coordinates in input CSV (default: 640)")
    parser.add_argument("--iou_thr", type=float, default=0.5, help="WBF IoU threshold (default: 0.5)")
    
    args = parser.parse_args()
    
    fuse_annotations(args.csv, args.output, args.target_size, args.input_scale, args.iou_thr)
    
    if args.split_dir is not None:
        split_and_save(args.output, args.split_dir)
