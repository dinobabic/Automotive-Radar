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
    batch_size = 64

    radar_dataset = RadarDataset(train_dataset_dir)
    radar_dataset = Subset(radar_dataset, list(range(3000)))

    dataset = RadarBEVDataset(radar_dataset, aggregate=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = NVRadarNet(in_channels=5, num_classes=2).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)

    criterion_seg = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 5.0]).to(device)) # higher weight to class 1 (car)
    criterion_reg = nn.L1Loss(reduction='none')

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        batch_count = 0

        for (bev_images, seg_target, reg_target, reg_mask) in tqdm(loader, desc=f"Epoch {epoch}, Lr: {optimizer.param_groups[0]['lr']:.1e}"):
            bev_images = bev_images.to(device)
            seg_target = seg_target.to(device)
            reg_target = reg_target.to(device)
            reg_mask = reg_mask.to(device)

            seg_logits, reg_pred = model(bev_images)

            seg_loss = criterion_seg(seg_logits, seg_target)
            reg_loss = criterion_reg(reg_pred, reg_target)

            reg_mask = reg_mask.unsqueeze(1) 
            reg_loss = reg_loss * reg_mask
            reg_loss = reg_loss.sum() / (reg_mask.sum() + 1e-6)

            total_loss = seg_loss + reg_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()
            batch_count += 1
        
        epoch_loss /= batch_count
        torch.save(model.state_dict(), f"./experiments/models/nv_radar_net_epoch_{epoch}.pth")
        lr_scheduler.step()
        
        validation_loss = validate_radar_detection_model(validation_dataset_dir, device)
        print(f"Epoch {epoch}: Train Loss = {epoch_loss:.4f}, Validation Loss = {validation_loss:.4f}")

def validate_radar_detection_model(dataset_dir, device):
    radar_dataset = RadarDataset(dataset_dir)
    dataset = RadarBEVDataset(radar_dataset, aggregate=True)
    loader = DataLoader(dataset, batch_size=64, shuffle=False)  

    model = NVRadarNet(in_channels=5, num_classes=2).to(device)
    model.load_state_dict(torch.load("nv_radar_net_epoch_43.pth", map_location='cpu'))
    model.eval()

    criterion_seg = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 5.0]).to(device)) # higher weight to class 1 (car)
    criterion_reg = nn.L1Loss(reduction='none')

    epoch_loss = 0.0
    batch_count = 0
    for (bev_images, seg_target, reg_target, reg_mask) in tqdm(loader, desc="Validation"):
        bev_images = bev_images.to(device)
        seg_target = seg_target.to(device)
        reg_target = reg_target.to(device)
        reg_mask = reg_mask.to(device)

        with torch.no_grad():
            seg_logits, reg_pred = model(bev_images)

        seg_loss = criterion_seg(seg_logits, seg_target)
        reg_loss = criterion_reg(reg_pred, reg_target)

        reg_mask = reg_mask.unsqueeze(1) 
        reg_loss = reg_loss * reg_mask
        reg_loss = reg_loss.sum() / (reg_mask.sum() + 1e-6)

        total_loss = seg_loss + reg_loss
        
        epoch_loss += total_loss.item()
        batch_count += 1

    epoch_loss /= batch_count
    return epoch_loss