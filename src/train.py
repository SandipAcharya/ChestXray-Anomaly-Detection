"""
CMRF-Net Core Training & Optimization Engine
Author: Antigravity AI Pair Programmer
"""

import os
import math
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Import local modules
from model import CMRFNet
from dataset import ChestXrayDataset, collate_fn
from loss import CMRFLoss
from utils import load_config, EMA, save_checkpoint, load_checkpoint

def set_seed(seed=42):
    """Set random seeds for deterministic reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_optimizer(model, base_lr, weight_decay):
    """
    Decoupled weight decay parameter grouping.
    Excludes biases and normalization layer weights from weight decay.
    """
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Exclude normalization layers and biases
        if len(param.shape) == 1 or name.endswith(".bias") or ".bn" in name or ".gn" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
            
    optimizer_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0}
    ]
    
    return torch.optim.AdamW(optimizer_groups, lr=base_lr, betas=(0.9, 0.999))

def train_epoch(model, dataloader, optimizer, loss_fn, scaler, device, ema, grad_clip):
    """Train the model for a single epoch."""
    model.train()
    total_loss = 0.0
    loss_components = {'box_loss': 0.0, 'obj_loss': 0.0, 'cls_loss': 0.0, 'triage_loss': 0.0}
    
    progress_bar = tqdm(dataloader, desc="Training Batches")
    for batch_idx, (images, targets, triage_targets) in enumerate(progress_bar):
        images = images.to(device)
        targets = targets.to(device)
        triage_targets = triage_targets.to(device)
        
        optimizer.zero_grad()
        
        # Native Automatic Mixed Precision (AMP)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            predictions = model(images)
            loss, components = loss_fn(predictions, targets, triage_targets)
            
        if scaler is not None:
            # Scaled backward pass
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            
        # Update Exponential Moving Average parameters
        if ema is not None:
            ema.update()
            
        total_loss += loss.item()
        for k in loss_components.keys():
            loss_components[k] += components[k]
            
        progress_bar.set_postfix(
            loss=loss.item(), 
            box=components['box_loss'], 
            obj=components['obj_loss']
        )
        
    num_batches = len(dataloader)
    epoch_loss = total_loss / num_batches
    avg_components = {k: v / num_batches for k, v in loss_components.items()}
    return epoch_loss, avg_components

@torch.no_grad()
def validate_epoch(model, dataloader, loss_fn, device, ema):
    """Validate model performance at end of epoch."""
    # Apply smoothed EMA parameters for evaluation
    if ema is not None:
        ema.apply_shadow()
        
    model.eval()
    total_loss = 0.0
    loss_components = {'box_loss': 0.0, 'obj_loss': 0.0, 'cls_loss': 0.0, 'triage_loss': 0.0}
    
    for images, targets, triage_targets in dataloader:
        images = images.to(device)
        targets = targets.to(device)
        triage_targets = triage_targets.to(device)
        
        # Forward pass
        predictions = model(images)
        loss, components = loss_fn(predictions, targets, triage_targets)
        
        total_loss += loss.item()
        for k in loss_components.keys():
            loss_components[k] += components[k]
            
    # Restore original training weights
    if ema is not None:
        ema.restore()
        
    num_batches = len(dataloader)
    epoch_loss = total_loss / num_batches
    avg_components = {k: v / num_batches for k, v in loss_components.items()}
    return epoch_loss, avg_components

def main():
    parser = argparse.ArgumentParser(description="CMRF-Net Training Pipeline")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config file")
    parser.add_argument("--train_csv", type=str, required=True, help="Path to train_annotations.csv")
    parser.add_argument("--val_csv", type=str, required=True, help="Path to val_annotations.csv")
    parser.add_argument("--img_dir", type=str, required=True, help="Path to preprocessed images directory")
    parser.add_argument("--checkpoint_dir", type=str, default="runs/checkpoints", help="Where to save checkpoints")
    parser.add_argument("--log_dir", type=str, default="runs/logs", help="Tensorboard logging directory")
    parser.add_argument("--resume", type=str, default=None, help="Resume training from path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    set_seed(args.seed)
    
    # 1. Load Configurations
    config = load_config(args.config)
    input_size = config['data']['input_size']
    classes = config['data']['classes']
    num_classes = config['data']['num_classes']
    triage_classes = config['data']['triage_classes']
    
    train_cfg = config['train']
    loss_cfg = config['loss']
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Using active hardware: {device}")
    
    # 2. Instantiate Datasets & DataLoaders
    train_dataset = ChestXrayDataset(args.train_csv, args.img_dir, input_size, is_training=True, classes=classes)
    val_dataset = ChestXrayDataset(args.val_csv, args.img_dir, input_size, is_training=False, classes=classes)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=train_cfg['batch_size'], 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True, 
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=train_cfg['batch_size'], 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True, 
        collate_fn=collate_fn
    )
    
    # 3. Compile Model, Optimizer, Loss, and Scheduler
    model = CMRFNet(num_classes=num_classes, num_anchors=3, triage_classes=triage_classes).to(device)
    optimizer = get_optimizer(model, train_cfg['base_lr'], train_cfg['weight_decay'])
    loss_fn = CMRFLoss(num_classes=num_classes, triage_classes=triage_classes, config=loss_cfg)
    
    # Two-phase Learning Rate Schedule: Warmup (5 epochs) + Cosine Annealing (to 300 epochs)
    warmup_epochs = train_cfg['warmup_epochs']
    total_epochs = train_cfg['epochs']
    
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # Linear Warmup phase
            return float(epoch) / float(max(1, warmup_epochs))
        # Cosine Annealing decay phase
        progress = float(epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        return max(train_cfg['min_lr'] / train_cfg['base_lr'], 0.5 * (1.0 + math.cos(math.pi * progress)))
        
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # 4. Stabilization Mechanisms (EMA, GradScaler)
    ema = EMA(model, decay=train_cfg['ema_decay'])
    scaler = torch.cuda.amp.GradScaler() if train_cfg['mixed_precision'] else None
    
    # Checkpoint states
    start_epoch = 0
    best_val_loss = float('inf')
    
    # Optionally Resume Training
    if args.resume:
        start_epoch, best_val_loss = load_checkpoint(args.resume, model, optimizer, scheduler, ema)
        
    # Tensorboard Monitoring
    writer = SummaryWriter(log_dir=args.log_dir)
    
    # 5. Training loop
    early_stop_counter = 0
    patience = train_cfg['early_stopping_patience']
    
    print(f"🚀 Commencing training loop: {total_epochs} epochs.")
    for epoch in range(start_epoch, total_epochs):
        print(f"\n--- Epoch {epoch+1}/{total_epochs} ---")
        
        # Train
        train_loss, train_components = train_epoch(
            model, train_loader, optimizer, loss_fn, scaler, device, ema, train_cfg['grad_clip_norm']
        )
        # Validate
        val_loss, val_components = validate_epoch(model, val_loader, loss_fn, device, ema)
        
        # Step LR Scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"📈 Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Current LR: {current_lr:.6f}")
        
        # Log to TensorBoard
        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Val", val_loss, epoch)
        writer.add_scalar("LR/Current", current_lr, epoch)
        
        for k in train_components.keys():
            writer.add_scalar(f"LossComponents/Train/{k}", train_components[k], epoch)
            writer.add_scalar(f"LossComponents/Val/{k}", val_components[k], epoch)
            
        # Check Best Performance & Save
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            
        # Save Checkpoint
        state = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'ema_state_dict': ema.shadow,
            'best_val_loss': best_val_loss,
            'config': config
        }
        
        # Save regular checkpoint and check best
        save_checkpoint(state, is_best, args.checkpoint_dir, filename="checkpoint.pth")
        
        # Save periodic model every 10 epochs
        if (epoch + 1) % 10 == 0:
            save_checkpoint(state, False, args.checkpoint_dir, filename=f"checkpoint_epoch_{epoch+1}.pth")
            
        # Early Stopping check
        if early_stop_counter >= patience:
            print(f"🛑 Early stopping triggered at epoch {epoch+1} (No improvement for {patience} epochs).")
            break
            
    writer.close()
    print("🏁 Training complete!")

if __name__ == "__main__":
    main()
