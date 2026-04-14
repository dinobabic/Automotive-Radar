import numpy as np
import matplotlib.pyplot as plt
import os

from dataset import RadarDataset, RadarBEVDataset
from visualization import *
from radar_net import NVRadarNet

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F

from sklearn.cluster import DBSCAN

def visualize_radar_detection_model_results(dataset_dir, model_path, device, example_idx=10, save_dir="./experiments/visualization_results"):
    radar_dataset = RadarDataset(dataset_dir)

    dataset = RadarBEVDataset(radar_dataset, aggregate=True)

    model = NVRadarNet(in_channels=5, num_classes=2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    bev_images, seg_target, reg_target, reg_mask = dataset[example_idx]
    bev_images = bev_images.unsqueeze(0).to(device)
    seg_target = seg_target.unsqueeze(0).to(device)
    reg_target = reg_target.unsqueeze(0).to(device)
    reg_mask = reg_mask.unsqueeze(0).to(device)

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

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    plt.savefig(os.path.join(save_dir, "segmentation_output.png"))