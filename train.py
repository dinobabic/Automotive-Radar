import os
import numpy as np

from dataset import RadarDataset, RadarBEVDataset
from visualization import *
from radar_net import NVRadarNet

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F

from tqdm import tqdm

def train_radar_detection_model(train_dataset_dir, validation_dataset_dir, device):
    epochs = 50
    batch_size = 128

    radar_dataset = RadarDataset(train_dataset_dir)
    radar_dataset = Subset(radar_dataset, list(range(2000)))

    dataset = RadarBEVDataset(radar_dataset, aggregate=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = NVRadarNet(in_channels=5, num_classes=2).to(device)

    # separate log_var parameters from the rest of the parameters in the network
    base_params = [p for n, p in model.named_parameters() if 'log_var' not in n]
    loss_weight_params = [model.log_var_seg, model.log_var_reg]

    optimizer = torch.optim.Adam([
        {'params': base_params, 'lr': 1e-4},
        {'params': loss_weight_params, 'lr': 1e-2}  # 100x higher LR for rapid adaptation
    ])

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)

    criterion_seg = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 5.0]).to(device)) # higher weight to class 1 (car)
    criterion_reg = nn.L1Loss(reduction='none')

    if not os.path.exists("./experiments/training_visualization"): 
        os.makedirs("./experiments/training_visualization")

    total_train_loss_history, seg_train_loss_history, reg_train_loss_history = [], [], []
    total_val_loss_history, seg_val_loss_history, reg_val_loss_history = [], [], []
    learned_weights_history = []

    for epoch in range(epochs):
        model.train()
        epoch_total_loss, epoch_seg_loss, epoch_reg_loss = 0.0, 0.0, 0.0
        batch_count = 0

        for (bev_images, seg_target, reg_target, reg_mask) in tqdm(loader, desc=f"Epoch {epoch}, Lr: {optimizer.param_groups[0]['lr']:.1e}"):
            bev_images = bev_images.to(device)
            seg_target = seg_target.to(device)
            reg_target = reg_target.to(device)
            reg_mask = reg_mask.to(device)

            seg_logits, reg_pred = model(bev_images)

            log_var_seg = model.log_var_seg
            log_var_reg = model.log_var_reg

            seg_loss_raw = criterion_seg(seg_logits, seg_target)
            seg_loss = torch.exp(-log_var_seg) * seg_loss_raw + log_var_seg

            reg_loss_raw = criterion_reg(reg_pred, reg_target)
            reg_mask = reg_mask.unsqueeze(1) 
            reg_loss = reg_loss_raw * reg_mask
            reg_loss = reg_loss.sum() / (reg_mask.sum() * reg_pred.shape[1] + 1e-6)
            reg_loss = torch.exp(-log_var_reg) * reg_loss + log_var_reg

            total_loss = seg_loss + reg_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            epoch_seg_loss += seg_loss_raw.item()
            epoch_reg_loss += reg_loss_raw.item()
            epoch_total_loss += total_loss.item()
            batch_count += 1
        
        epoch_total_loss /= batch_count
        epoch_seg_loss /= batch_count
        epoch_reg_loss /= batch_count

        total_train_loss_history.append(epoch_total_loss)
        seg_train_loss_history.append(epoch_seg_loss)
        reg_train_loss_history.append(epoch_reg_loss)

        learned_weights_history.append((torch.exp(-model.log_var_seg).item(), torch.exp(-model.log_var_reg).item()))

        lr_scheduler.step()

        if os.path.exists("./experiments/models") == False:
            os.makedirs("./experiments/models")

        model_path = f"./experiments/models/nv_radar_net_epoch_{epoch}.pth" 
        torch.save(model.state_dict(), model_path)

        total_val_loss, seg_val_loss, reg_val_loss = validate_radar_detection_model(validation_dataset_dir, model_path, device)
        total_val_loss_history.append(total_val_loss)
        seg_val_loss_history.append(seg_val_loss)
        reg_val_loss_history.append(reg_val_loss)

        print(f"Epoch {epoch}: Total Train Loss = {epoch_total_loss:.4f}, Total Validation Loss = {total_val_loss:.4f} \
                Segmentation Train Loss = {epoch_seg_loss:.4f}, Segmentation Validation Loss = {seg_val_loss:.4f} \
                Regression Train Loss = {epoch_reg_loss:.4f}, Regression Validation Loss = {reg_val_loss:.4f} \
                Learned Weights Segmentation Loss = {torch.exp(-model.log_var_seg).item()} \
                Learned Weights Regression Loss = {torch.exp(-model.log_var_reg).item()}")


        plt.figure(figsize=(10, 5))
        plt.plot(total_train_loss_history, label='Train Loss')
        plt.plot(total_val_loss_history, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Total Training and Validation Loss Over Epochs')
        plt.legend()
        plt.grid()
        plt.savefig("./experiments/training_visualization/total_train_loss_curve.png")
        plt.close()

        plt.figure(figsize=(10, 5))
        plt.plot(seg_train_loss_history, label='Train Loss')
        plt.plot(seg_val_loss_history, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Segmentation Training and Validation Loss Over Epochs')
        plt.legend()
        plt.grid()
        plt.savefig("./experiments/training_visualization/seg_train_loss_curve.png")
        plt.close()

        plt.figure(figsize=(10, 5))
        plt.plot(reg_train_loss_history, label='Train Loss')
        plt.plot(reg_val_loss_history, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Regression Training and Validation Loss Over Epochs')
        plt.legend()
        plt.grid()
        plt.savefig("./experiments/training_visualization/reg_train_loss_curve.png")
        plt.close()

        plt.figure(figsize=(10, 5))
        plt.plot([w[0] for w in learned_weights_history], label='Segmentation Loss Weight')
        plt.plot([w[1] for w in learned_weights_history], label='Regression Loss Weight')
        plt.xlabel('Epoch')
        plt.ylabel('Learned Weight')
        plt.title('Learned Loss Weights Over Epochs')
        plt.legend()
        plt.grid()
        plt.savefig("./experiments/training_visualization/learned_loss_weights.png")
        plt.close()

def validate_radar_detection_model(dataset_dir, model_path, device):
    radar_dataset = RadarDataset(dataset_dir)
    dataset = RadarBEVDataset(radar_dataset, aggregate=True)    
    dataset = Subset(dataset, list(range(500)))
    loader = DataLoader(dataset, batch_size=128, shuffle=False)

    model = NVRadarNet(in_channels=5, num_classes=2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    criterion_seg = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 5.0]).to(device)) # higher weight to class 1 (car)
    criterion_reg = nn.L1Loss(reduction='none')

    total_epoch_loss, seg_epoch_loss, reg_epoch_loss = 0.0, 0.0, 0.0
    batch_count = 0
    for (bev_images, seg_target, reg_target, reg_mask) in tqdm(loader, desc="Validation"):
        bev_images = bev_images.to(device)
        seg_target = seg_target.to(device)
        reg_target = reg_target.to(device)
        reg_mask = reg_mask.to(device)

        with torch.no_grad():
            seg_logits, reg_pred = model(bev_images)

        log_var_seg = model.log_var_seg
        log_var_reg = model.log_var_reg

        seg_loss_raw = criterion_seg(seg_logits, seg_target)
        seg_loss = torch.exp(-log_var_seg) * seg_loss_raw + log_var_seg

        reg_loss_raw = criterion_reg(reg_pred, reg_target)
        reg_mask = reg_mask.unsqueeze(1) 
        reg_loss = reg_loss_raw * reg_mask
        reg_loss = reg_loss.sum() / (reg_mask.sum() * reg_pred.shape[1] + 1e-6)
        reg_loss = torch.exp(-log_var_reg) * reg_loss + log_var_reg

        total_loss = seg_loss + reg_loss
        
        seg_epoch_loss += seg_loss_raw.item()
        reg_epoch_loss += reg_loss_raw.item()
        total_epoch_loss += total_loss.item()

        batch_count += 1

    total_epoch_loss /= batch_count
    seg_epoch_loss /= batch_count
    reg_epoch_loss /= batch_count

    return total_epoch_loss, seg_epoch_loss, reg_epoch_loss