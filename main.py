import numpy as np
import argparse
import matplotlib.pyplot as plt

from dataset import RadarDataset, RadarBEVDataset
from visualization import *
from radar_net import NVRadarNet

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F

from tqdm import tqdm

from sklearn.cluster import DBSCAN

root_radar_dir = '/Volumes/T7/lrrr_sim_data/radar_data'
root_image_dir = '/Volumes/T7/lrrr_sim_data/image_data'

device = 'cpu'

def test(root_radar_dir):
    radar_dataset = RadarDataset(root_radar_dir)

    dataset = RadarBEVDataset(radar_dataset, aggregate=True)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = NVRadarNet(in_channels=5, num_classes=2).to(device)
    model.load_state_dict(torch.load("nv_radar_net_epoch_43.pth", map_location='cpu'))
    model.eval()

    bev_images, seg_target, reg_target, reg_mask = dataset[10]
    bev_images = bev_images.unsqueeze(0).to(device)
    seg_target = seg_target.unsqueeze(0).to(device)
    reg_target = reg_target.unsqueeze(0).to(device)
    reg_mask = reg_mask.unsqueeze(0).to(device)


    #import ipdb; ipdb.set_trace()
    with torch.no_grad():
        seg_logits, reg_pred = model(bev_images)
        seg_probs = F.softmax(seg_logits, dim=1)

    car_probs = seg_probs[0, 1]
    mask = car_probs > 0.5
    mask = mask.cpu().numpy()

    reg_pred = reg_pred.squeeze(0).cpu().numpy()

    ys, xs = np.where(mask)

    centers = []
    for y, x in zip(ys, xs):
        dx = reg_pred[0, y, x]
        dy = reg_pred[1, y, x]

        cx = x + 0.5 + dx
        cy = y + 0.5 + dy

        centers.append((cx, cy))
    
    centers = np.array(centers)

    clustering_alg = DBSCAN(eps=1.0, min_samples=3)
    cluster_labels = clustering_alg.fit_predict(centers)

    pred_bboxes = []

    for label in np.unique(cluster_labels):
        if label == -1:
            continue

        cluster_points = centers[cluster_labels == label]
        #plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {label}")
        cluster_indices = np.where(cluster_labels == label)[0]

        cx = np.mean(cluster_points[:, 0])
        cy = np.mean(cluster_points[:, 1])

        ws, ls, sins, coss = [], [], [], []

        for idx in cluster_indices:
            y = ys[idx]
            x = xs[idx]

            ws.append(reg_pred[2, y, x])
            ls.append(reg_pred[3, y, x])
            sins.append(reg_pred[4, y, x])
            coss.append(reg_pred[5, y, x])

        w = np.mean(ws)
        l = np.mean(ls)
        theta = np.arctan2(np.mean(sins), np.mean(coss))

        pred_bboxes.append((cx, cy, w, l, theta))

    # visualize results
    seg_probs = seg_probs.squeeze(0).cpu().numpy()
    fig, ax = plt.subplots()
    plt.imshow(seg_probs[1], cmap='gray', interpolation='nearest')

    for i in range(len(pred_bboxes)):
        cx, cy, w, l, theta = pred_bboxes[i]

        rect = plt.Rectangle(
            (cx - w/2, cy - l/2), 
            w, 
            l, 
            angle=np.rad2deg(theta), 
            rotation_point='center',
            edgecolor='red', 
            facecolor='none'
        )

        ax.add_patch(rect)

    plt.savefig("segmentation_output.png")

def train(root_radar_dir):
    epochs = 50
    batch_size = 64

    radar_dataset = RadarDataset(root_radar_dir)
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
        
        torch.save(model.state_dict(), f"nv_radar_net_epoch_{epoch}.pth")
        lr_scheduler.step()
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
    
    #train(args.root_radar_dir)
    test(args.root_radar_dir)


if __name__ == "__main__": 
    main()