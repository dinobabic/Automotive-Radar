import os
import numpy as np
import argparse
import matplotlib.pyplot as plt

from dataset import RadarDataset, RadarBEVDataset
from visualization import *
from radar_net import NVRadarNet
from train import train_radar_detection_model
from test import visualize_radar_detection_model_results

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F

from tqdm import tqdm

from sklearn.cluster import DBSCAN

device = 'cuda:1'

def main():
    parser = argparse.ArgumentParser(description="Automotive Radar Program")
    
    parser.add_argument("--root_radar_dir", help="Path to the root dir contaning train and test folders with radar data")
    
    args = parser.parse_args()
    root_radar_dir = args.root_radar_dir
    train_dataset_dir = os.path.join(root_radar_dir, "train_dataset")
    validation_dataset_dir = os.path.join(root_radar_dir, "test_dataset")
    
    #radar_dataset = RadarDataset(train_dataset_dir)    
    #visualize_radar_pcl(radar_dataset[10])
    #visualize_radar_pcl_aggregated_standard(radar_dataset, 10, 0.5)
    #visualize_radar_pcl_aggregated_fixed(radar_dataset, 0, 15)
    #visualize_radar_pcl_aggregated_dopp_drive(radar_dataset, current_frame)
    
    #train_radar_detection_model(train_dataset_dir, validation_dataset_dir, device)
    visualize_radar_detection_model_results(train_dataset_dir, "./experiments/models/nv_radar_net_epoch_12.pth", 'cuda:2', example_idx=4200, save_dir="./experiments/visualization_results")


if __name__ == "__main__": 
    main()