import numpy as np
import argparse

from dataset import RadarDataset, RadarBEVDataset
from visualization import *
from radar_net import NVRadarNet

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

from tqdm import tqdm

root_radar_dir = '/Volumes/T7/lrrr_sim_data/radar_data'
root_image_dir = '/Volumes/T7/lrrr_sim_data/image_data'

device = 'cpu'

def train(root_radar_dir):
    epochs = 50
    batch_size = 16

    radar_dataset = RadarDataset(root_radar_dir)
    
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

        for (bev_images, seg_target, reg_target, reg_mask) in tqdm(loader, desc=f"Epoch {epoch}"):
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
        
        print(f"Epoch {epoch}: Loss = {epoch_loss:.4f}")

def main():
    parser = argparse.ArgumentParser(description="Automotive Radar Program")
    
    parser.add_argument("--root_radar_dir", help="Path to the root dir contaning train and test folders with radar data")
    parser.add_argument("--mode", choices=['TRAIN', 'TEST', 'VISUALIZE'], help='Mode in which program will run. Possible values: TRAIN, TEST, VISUALIZE')
    
    args = parser.parse_args()
    
    radar_dataset = RadarDataset(args.root_radar_dir)
        
    #visualize_radar_pcl(radar_dataset[850])
    #visualize_radar_pcl_aggregated_standard(radar_dataset, 840, 0.5)
    #visualize_radar_pcl_aggregated_fixed(radar_dataset, 0, 15)
    #visualize_radar_pcl_aggregated_dopp_drive(radar_dataset, current_frame)
    
    train(args.root_radar_dir)


if __name__ == "__main__": 
    main()