import numpy as np
import argparse

from dataset import RadarDataset
from visualization import *

root_radar_dir = '/Volumes/T7/lrrr_sim_data/radar_data'
root_image_dir = '/Volumes/T7/lrrr_sim_data/image_data'

def main():
    parser = argparse.ArgumentParser(description="Automotive Radar Program")
    
    parser.add_argument("--root_radar_dir", help="Path to the root dir contaning train and test folders with radar data")
    parser.add_argument("--mode", choices=['TRAIN', 'TEST', 'VISUALIZE'], help='Mode in which program will run. Possible values: TRAIN, TEST, VISUALIZE')
    
    args = parser.parse_args()
    
    radar_dataset = RadarDataset(args.root_radar_dir)
        
    visualize_radar_pcl(radar_dataset[0])
    visualize_radar_pcl_aggregated_standard(radar_dataset, 0, 1.5)
    visualize_radar_pcl_aggregated_fixed(radar_dataset, 0, 10)
    #visualize_radar_pcl_aggregated_dopp_drive(radar_dataset, current_frame)
    


if __name__ == "__main__": 
    main()