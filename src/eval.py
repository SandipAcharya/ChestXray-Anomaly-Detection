"""
SOTA Clinical Evaluation & Grad-CAM Explainability Module
Author: Antigravity AI Pair Programmer
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import cv2
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, precision_recall_curve, confusion_matrix
from collections import Counter, defaultdict

# Import local modules
from model import CMRFNet
from dataset import ChestXrayDataset, collate_fn
from utils import load_config, non_maximum_suppression, draw_predictions, xywh2xyxy, box_iou

class GradCAM:
    """
    Grad-CAM class to extract and overlay visual attention heatmaps 
    for model transparency and radiologist trust.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_layers()

    def hook_layers(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        # Register hooks
        self.target_layer.register_forward_hook(forward_hook)
        # Using register_full_backward_hook to avoid deprecation warnings
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_heatmap(self, x, class_idx, triage_prediction):
        """
        Generate raw attention heatmap for the designated clinical category.
        """
        self.model.zero_grad()
        # Backpropagate gradients of designated clinical category
        triage_prediction[0, class_idx].backward(retain_graph=True)
        
        # Global Average Pooling (GAP) of feature map gradients
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        
        # Weighted sum of feature map activations
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F_relu = torch.clamp(cam, min=0) # Apply ReLU
        
        # Normalize heatmap to [0, 1]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-16)
        
        # Rescale to raw image resolution (512x512)
        heatmap = cam.squeeze().cpu().numpy()
        heatmap = cv2.resize(heatmap, (x.shape[3], x.shape[2]))
        return heatmap

def overlay_heatmap(image_rgb: np.ndarray, heatmap: np.ndarray, alpha=0.4) -> np.ndarray:
    """Overlay colorized jet heatmap on input grayscale image."""
    # Convert heatmap to uint8 color map
    heatmap_color = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    
    # Overlay onto input canvas
    overlayed = cv2.addWeighted(image_rgb, 1.0 - alpha, heatmap_color, alpha, 0)
    return overlayed

def calculate_detection_metrics(all_predictions, all_targets, num_classes, iou_threshold=0.5):
    """
    Compute rigorous object detection metrics (Precision, Recall, F1, mAP) on the test split.
    all_predictions: List of bounding box detections [N, 6] -> columns [xmin, ymin, xmax, ymax, conf, class_id]
    all_targets: List of ground-truth boxes [M, 5] -> columns [class_id, xcenter, ycenter, w, h] in normalized coordinates
    """
    class_tps = defaultdict(list)
    class_confs = defaultdict(list)
    class_num_gts = Counter()
    
    # Count ground-truths per class
    for img_idx, gt_boxes in enumerate(all_targets):
        if gt_boxes.size(0) == 0:
            continue
        for box in gt_boxes:
            c_id = int(box[0].item())
            class_num_gts[c_id] += 1
            
    # Match detections and ground-truths
    for img_idx, (preds, gt_boxes) in enumerate(zip(all_predictions, all_targets)):
        if preds.size(0) == 0:
            continue
            
        # Parse detections and sort by confidence
        preds = preds.cpu().numpy()
        preds = preds[np.argsort(preds[:, 4])[::-1]] # high conf first
        
        # Parse targets and scale to 512 grid size
        if gt_boxes.size(0) > 0:
            gt_boxes_xyxy = xywh2xyxy(gt_boxes[:, 1:]).cpu().numpy() * 512.0
            gt_classes = gt_boxes[:, 0].cpu().numpy().astype(int)
            matched_gts = set()
        else:
            gt_boxes_xyxy = np.empty((0, 4))
            gt_classes = np.array([])
            matched_gts = set()
            
        for det in preds:
            det_box = det[:4]
            det_conf = det[4]
            det_cls = int(det[5])
            
            class_confs[det_cls].append(det_conf)
            
            # Find overlaps with ground-truth of same class
            best_iou = 0.0
            best_gt_idx = -1
            
            for gt_i, (gt_box, gt_cls) in enumerate(zip(gt_boxes_xyxy, gt_classes)):
                if gt_cls != det_cls or gt_i in matched_gts:
                    continue
                # Calculate box IoU
                inter_x1 = max(det_box[0], gt_box[0])
                inter_y1 = max(det_box[1], gt_box[1])
                inter_x2 = min(det_box[2], gt_box[2])
                inter_y2 = min(det_box[3], gt_box[3])
                
                inter_w = max(0.0, inter_x2 - inter_x1)
                inter_h = max(0.0, inter_y2 - inter_y1)
                inter = inter_w * inter_h
                
                union = (det_box[2] - det_box[0]) * (det_box[3] - det_box[1]) + \
                        (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1]) - inter + 1e-16
                iou = inter / union
                
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_i
                    
            if best_iou >= iou_threshold:
                class_tps[det_cls].append(1.0)
                matched_gts.add(best_gt_idx)
            else:
                class_tps[det_cls].append(0.0)
                
    # Calculate Precision, Recall, AP per class
    ap_per_class = {}
    f1_per_class = {}
    p_per_class = {}
    r_per_class = {}
    
    for c_id in range(num_classes):
        num_gts = class_num_gts[c_id]
        tps = np.array(class_tps[c_id])
        confs = np.array(class_confs[c_id])
        
        if num_gts == 0:
            ap_per_class[c_id] = 0.0
            f1_per_class[c_id] = 0.0
            p_per_class[c_id] = 0.0
            r_per_class[c_id] = 0.0
            continue
            
        if len(tps) == 0:
            ap_per_class[c_id] = 0.0
            f1_per_class[c_id] = 0.0
            p_per_class[c_id] = 0.0
            r_per_class[c_id] = 0.0
            continue
            
        # Sort by confidence descending
        sort_indices = np.argsort(confs)[::-1]
        tps = tps[sort_indices]
        
        cum_tps = np.cumsum(tps)
        cum_fps = np.cumsum(1.0 - tps)
        
        precisions = cum_tps / (cum_tps + cum_fps + 1e-16)
        recalls = cum_tps / num_gts
        
        # Calculate AP (11-point interpolation or AUC PR curve)
        ap = 0.0
        for t in np.arange(0.0, 1.1, 0.1):
            prec_at_t = precisions[recalls >= t]
            if len(prec_at_t) > 0:
                ap += np.max(prec_at_t)
        ap_per_class[c_id] = ap / 11.0
        
        # Max F1 score
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-16)
        best_idx = np.argmax(f1_scores)
        f1_per_class[c_id] = f1_scores[best_idx]
        p_per_class[c_id] = precisions[best_idx]
        r_per_class[c_id] = recalls[best_idx]
        
    mean_ap = np.mean(list(ap_per_class.values()))
    return mean_ap, ap_per_class, f1_per_class, p_per_class, r_per_class

def calculate_classification_metrics(all_triage_preds, all_triage_targets, triage_classes):
    """
    Calculate diagnostic multi-label metrics (Sensitivity, Specificity, PPV, NPV, AUC-ROC).
    """
    preds = np.array(all_triage_preds)      # [N_samples, 15]
    targets = np.array(all_triage_targets)  # [N_samples, 15]
    
    auc_scores = {}
    class_metrics = {}
    
    for c_idx in range(triage_classes):
        y_true = targets[:, c_idx]
        y_score = preds[:, c_idx]
        
        # AUC-ROC calculation
        if len(np.unique(y_true)) > 1:
            auc = roc_auc_score(y_true, y_score)
        else:
            auc = 0.5
        auc_scores[c_idx] = auc
        
        # Binary confusion matrix for standard metrics (threshold at 0.5)
        y_pred = (y_score >= 0.5).astype(float)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        
        sensitivity = tp / (tp + fn + 1e-16)
        specificity = tn / (tn + fp + 1e-16)
        ppv = tp / (tp + fp + 1e-16)          # Positive Predictive Value
        npv = tn / (tn + fn + 1e-16)          # Negative Predictive Value
        accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-16)
        
        class_metrics[c_idx] = {
            'sensitivity': sensitivity,
            'specificity': specificity,
            'ppv': ppv,
            'npv': npv,
            'accuracy': accuracy,
            'auc': auc
        }
        
    return auc_scores, class_metrics

def main():
    parser = argparse.ArgumentParser(description="CMRF-Net SOTA Evaluation Pipeline")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config file")
    parser.add_argument("--test_csv", type=str, required=True, help="Path to test_annotations.csv")
    parser.add_argument("--img_dir", type=str, required=True, help="Path to preprocessed images directory")
    parser.add_argument("--weights", type=str, required=True, help="Path to best_model.pth weight file")
    parser.add_argument("--gradcam_out", type=str, default="runs/explainability", help="Where to save Grad-CAM overlays")
    parser.add_argument("--visualize_count", type=int, default=5, help="Number of images to generate Grad-CAM overlays for")
    
    args = parser.parse_args()
    
    # Load Configurations
    config = load_config(args.config)
    input_size = config['data']['input_size']
    classes = config['data']['classes']
    num_classes = config['data']['num_classes']
    triage_classes = config['data']['triage_classes']
    triage_class_names = classes + ["No finding"]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Loading model weights onto device: {device}")
    
    # Load test dataset
    test_dataset = ChestXrayDataset(args.test_csv, args.img_dir, input_size, is_training=False, classes=classes)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=2, collate_fn=collate_fn)
    
    # Compile model and load weights
    model = CMRFNet(num_classes=num_classes, num_anchors=3, triage_classes=triage_classes).to(device)
    checkpoint = torch.load(args.weights, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Initialize Lists to accumulate metrics
    all_detections = []
    all_targets = []
    all_triage_preds = []
    all_triage_targets = []
    
    # Standard forward evaluation loops
    print("📈 Running forward evaluation splits...")
    for batch_idx, (images, targets, triage_targets) in enumerate(tqdm(test_loader, desc="Evaluation Batches")):
        images = images.to(device)
        
        with torch.no_grad():
            pred_high, pred_med, pred_low, global_triage = model(images)
            
        # Parse variable-length targets back into per-image targets list
        for b_idx in range(images.size(0)):
            img_gts = targets[targets[:, 0] == b_idx]
            all_targets.append(img_gts[:, 1:]) # Store class_id and normalized coord
            
        # Run NMS post-processing on scale predictions
        # non_maximum_suppression requires predictions as list of scales
        batch_detections = non_maximum_suppression(
            [pred_high, pred_med, pred_low], 
            conf_thres=0.15, 
            iou_thres=0.45
        )
        all_detections.extend(batch_detections)
        
        all_triage_preds.extend(global_triage.cpu().numpy())
        all_triage_targets.extend(triage_targets.numpy())
        
    # Calculate Regional Object Detection metrics
    mAP, class_aps, class_f1s, class_prec, class_rec = calculate_detection_metrics(
        all_detections, all_targets, num_classes, iou_threshold=0.5
    )
    
    # Calculate Global Clinical Triage Classification metrics
    auc_scores, class_metrics = calculate_classification_metrics(
        all_triage_preds, all_triage_targets, triage_classes
    )
    
    # Output detailed SOTA clinical summary report
    print("\n" + "="*50)
    print("    SOTA CLINICAL PERFORMANCE REPORT - CMRF-Net")
    print("="*50)
    print(f"Overall Box mAP@0.5: {mAP:.4f}")
    
    print("\nRegional Diagnostic Detection Metrics per Pathology:")
    print(f"{'Pathology Class':<22} | {'AP@0.5':<8} | {'Precision':<9} | {'Recall':<8} | {'F1-Score':<8}")
    print("-"*64)
    for c_id in range(num_classes):
        print(f"{classes[c_id]:<22} | {class_aps[c_id]:.4f} | {class_prec[c_id]:.4f} | {class_rec[c_id]:.4f} | {class_f1s[c_id]:.4f}")
        
    print("\nAuxiliary Image-Level Triage Classification Metrics:")
    print(f"{'Clinical Category':<22} | {'AUC-ROC':<8} | {'Sensitivity':<11} | {'Specificity':<11} | {'NPV':<8}")
    print("-"*69)
    for c_idx in range(triage_classes):
        metrics = class_metrics[c_idx]
        print(f"{triage_class_names[c_idx]:<22} | {metrics['auc']:.4f} | {metrics['sensitivity']:.4f} | {metrics['specificity']:.4f} | {metrics['npv']:.4f}")
    print("="*50)
    
    # 6. EXPLAINABILITY: Generate Visual Attention overlays (Grad-CAM)
    # Hook the bottleneck layer of the Deep Recombination Cluster (Stage 9 ASPP projection block)
    target_layer = model.stage9.aspp.project
    gradcam = GradCAM(model, target_layer)
    
    # Select visual sample images
    os.makedirs(args.gradcam_out, exist_ok=True)
    print(f"👁️ Generating {args.visualize_count} explainability attention maps...")
    
    for i in range(min(args.visualize_count, len(test_dataset))):
        img_tensor, _, triage_target = test_dataset[i]
        
        # Unsqueeze to add batch dimension [1, 3, 512, 512]
        x = img_tensor.unsqueeze(0).to(device)
        x.requires_grad = True
        
        # Forward pass to trigger hooks
        pred_high, pred_med, pred_low, global_triage = model(x)
        
        # Check active pathologies present in the ground-truth
        active_ids = torch.where(triage_target == 1.0)[0].tolist()
        if not active_ids:
            active_ids = [14] # Fallback to No finding
            
        for class_idx in active_ids:
            heatmap = gradcam.generate_heatmap(x, class_idx, global_triage)
            
            # De-normalize image tensor for plotting
            img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
            # ImageNet de-normalization
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img_np = std * img_np + mean
            img_np = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)
            
            # Create color overlay
            overlayed = overlay_heatmap(img_np, heatmap, alpha=0.45)
            
            # Save visual plots side-by-side
            fig, ax = plt.subplots(1, 2, figsize=(12, 6))
            ax[0].imshow(img_np)
            ax[0].set_title("Grayscale Radiography")
            ax[0].axis('off')
            
            ax[1].imshow(overlayed)
            ax[1].set_title(f"Visual Attention: {triage_class_names[class_idx]}")
            ax[1].axis('off')
            
            out_fn = os.path.join(args.gradcam_out, f"gradcam_sample_{i}_{triage_class_names[class_idx].replace('/', '_').replace(' ', '_')}.png")
            plt.savefig(out_fn, bbox_inches="tight", dpi=150)
            plt.close()
            print(f"💾 Saved visual overlay to: {out_fn}")
            
    print("🏁 Evaluation complete!")

if __name__ == "__main__":
    main()
