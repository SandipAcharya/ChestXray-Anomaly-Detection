import os
import sys
import torch
import argparse
import json
from pathlib import Path
from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.torch_utils import select_device
from utils.datasets import LoadImages
from utils.plots import plot_one_box
import cv2

# Define a list of dark colors (dark red, dark blue, dark green, etc.)
dark_colors = [
    (33, 33, 33),  # Dark gray
    (139, 0, 0),   # Dark red
    (0, 0, 139),   # Dark blue
    (75, 0, 130),  # Dark violet
    (139, 69, 19), # Dark brown
    (0, 128, 128), # Dark teal
    (128, 0, 0),   # Dark maroon
    (0, 0, 128),   # Dark navy
    (255, 69, 0),  # Dark orange
    (128, 0, 128)  # Dark purple
]

def plot_one_box(xyxy, im0, label=None, color=None, line_thickness=3):
    # Draw a box with a given color
    if color is None:
        color = (33, 33, 33)  # Default to dark gray if no color provided

    # Draw rectangle with a thicker line
    cv2.rectangle(im0, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), color, line_thickness * 2)

    if label:
        # Get the size of the label
        label_size = cv2.getTextSize(label, 0, 0.5, 1)[0]
        label_x1 = int(xyxy[0])
        label_y1 = int(xyxy[1]) - label_size[1] - 3
        label_x2 = label_x1 + label_size[0]
        label_y2 = label_y1 + label_size[1] + 3

        # Draw background for label (darkened box)
        label_bg_color = (0, 0, 0)  # Dark background for label
        cv2.rectangle(im0, (label_x1, label_y1), (label_x2, label_y2), label_bg_color, -1)

        # Add the label text (white)
        cv2.putText(im0, label, (label_x1, label_y1 + label_size[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

def detect(source='data/images', weights='yolov7.pt', imgsz=640, conf_thres=0.25, iou_thres=0.45, 
           project='processed_images', name=None, exist_ok=False):
    # Initialize
    device = select_device('')
    model = attempt_load(weights, map_location=device)
    stride = int(model.stride.max())
    imgsz = check_img_size(imgsz, s=stride)
    
    # Create output directory structure
    project = Path(project)
    if name is None:
        name = Path(source).stem  # Use source filename if no name provided
    save_dir = project
    
    dataset = LoadImages(source, img_size=imgsz, stride=stride)
    
    anomaly_results = []
    colors = {}  # Store unique colors for each anomaly
    
    def get_color(cls_id):
        if cls_id not in colors:
            colors[cls_id] = dark_colors[cls_id % len(dark_colors)]  # Assign a dark color from the list
        return colors[cls_id]
    
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.float() / 255.0  # Normalize
        img = img.unsqueeze(0) if img.ndimension() == 3 else img

        # Inference
        pred = model(img, augment=False)[0]
        
        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres)

        # Process detections
        for det in pred:
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()
                
                # Process each detection
                for *xyxy, conf, cls in det:
                    label = f"{model.names[int(cls)]} {conf:.2f}"
                    print(f"Detected: {label}")
                    
                    # Get unique color for each anomaly
                    color = get_color(int(cls))
                    plot_one_box(xyxy, im0s, label=label, color=color, line_thickness=1)  # Adjust line thickness as needed
                    
                    # Add to results
                    anomaly_results.append({
                        'anomalyName': model.names[int(cls)], 
                        'percentage': f"{conf:.2%}"
                    })
            
            # Save processed image
            output_filename = Path(path).name
            save_path = save_dir / output_filename
            cv2.imwrite(str(save_path), im0s)
            print(f"Saved processed image to: {save_path}")
    
    # Save anomaly results to JSON
    json_path = save_dir / 'anomalies.json'
    with open(json_path, 'w') as f:
        json.dump(anomaly_results, f, indent=4)
    print(f"Detection results saved in: {json_path}")
    
    return {
        'processed_image': str(save_path) if 'save_path' in locals() else None,
        'anomalies': anomaly_results
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='image1.png', help='image source path')
    parser.add_argument('--weights', type=str, default='xray.pt', help='model weights')
    parser.add_argument('--img-size', type=int, default=640, help='image size')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold')
    parser.add_argument('--project', type=str, default='processed_images', help='output directory')
    parser.add_argument('--name', type=str, default=None, help='output subdirectory name')
    parser.add_argument('--exist-ok', action='store_true', help='existing output dir ok, do not increment')
    args = parser.parse_args()
    
    results = detect(
        args.source, 
        args.weights, 
        args.img_size, 
        args.conf_thres, 
        args.iou_thres,
        args.project,
        args.name,
        args.exist_ok
    )
    
    print("Detection complete!")
    if results['processed_image']:
        print(f"Final processed image: {results['processed_image']}")
    print(f"Found {len(results['anomalies'])} anomalies")
