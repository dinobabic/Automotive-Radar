import numpy as np

from dataset import RadarDataset
from visualization import *

root_radar_dir = '/Volumes/T7/lrrr_sim_data/radar_data'
root_image_dir = '/Volumes/T7/lrrr_sim_data/image_data'

def main():
    radar_dataset = RadarDataset(root_radar_dir)
        
    visualize_radar_pcl(radar_dataset[0])
    visualize_radar_pcl_aggregated(radar_dataset, 0)
    


if __name__ == "__main__": 
    main()