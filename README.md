# Automotive Radar BEV Object Detection

This project implements bird's eye view (BEV) object detection using automotive radar data. The model uses a ResNet-based U-Net architecture to predict object locations and classes from radar point clouds. Model was trained on a synthetic dataset published in paper DoppDrive (https://arxiv.org/pdf/2508.12330) - find processing pipelin in dataset.py. 

Additionally, notebook tutorial_iq.ipynb implements full radar signal processing pipeline 

## Results

**Radar Point Cloud - single frame**
![Radar Point Cloud](experiments/visualization_results/radar_pcl_visualization.png)

**Aggregated Radar Point Cloud - aggregated for 0.5 seconds**
![Aggregated Radar PCL](experiments/visualization_results/radar_pcl_aggregated_visualization.png)

**BEV Visualization - projection of radar points in BEV together with the labels**
![BEV Visualization](experiments/visualization_results/bev_visualization.png)

**Segmentation Target - example target, input for the model**
![Segmentation Target](experiments/visualization_results/segmentation_target.png)

**Segmentation Output - output of the model**
![Segmentation Output](experiments/visualization_results/segmentation_output.png)
