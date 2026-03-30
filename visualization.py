import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from utils import transform_points_to_current_frame

def visualize_radar_pcl(radar_frame: dict):
    radar_pcl = radar_frame['radar_pts']
    pts_dynamic = np.abs(radar_frame['dynamic_doppler']) > 0.3

    fig, ax = plt.subplots()
    
    ax.scatter(radar_pcl[:,0][pts_dynamic], radar_pcl[:,1][pts_dynamic], color='blue', s=15) # dynamic points
    ax.scatter(radar_pcl[:,0][~pts_dynamic], radar_pcl[:,1][~pts_dynamic], color='orange', s=15) # static points
    
    for i in range(radar_frame['targets_bboxes'].shape[0]):
        # draw a bounding box around target
        cx, cy, _, ex, ey, _, yaw = radar_frame['targets_bboxes'][i]
        rect = Rectangle(
            xy=(cx-ex/2, cy-ey/2),
            width=ex,
            height=ey, 
            angle=yaw,
            rotation_point='center',
            edgecolor='red',
            facecolor='none',
            fill=False,
        )
        ax.add_patch(rect)

        # draw target's velocity vector
        vx, vy, _ = radar_frame['targets_vel'][i]
        plt.quiver(
            cx, cy,       
            vx, vy,
            angles='xy',
            scale_units='xy',
            scale=2.0,
            color='green'
        )

    ax.grid(True)
    ax.set_aspect('equal')
    plt.show()

def visualize_radar_pcl_aggregated_standard(radar_dataset, start_idx, aggregation_time_window=0.5):
    '''
        Function aggregates radar frames inside a 0.5 seconds window and visualizes 
        radar point cloud together with the labels from the current frame (latest aggregated frame).
    '''
    init_timestamp = radar_dataset[start_idx]['timestamp']
    idx = start_idx + 1
    while (True):
        next_timestamp = radar_dataset[idx]['timestamp']
        #print(f'init timestamp: {init_timestamp}, next timestamp: {next_timestamp}')
        if next_timestamp - init_timestamp < aggregation_time_window:
            idx += 1
        else:
            break
        
    fig, ax = plt.subplots()
    
    T_ego_local2global_current = radar_dataset[idx-1]['T_ego_local2global']
    T_ego_global2local_current = np.linalg.inv(T_ego_local2global_current)

    # draw radar points for every aggregated frame
    for k in range(start_idx, idx):
        radar_frame = radar_dataset[k]
        radar_pcl = radar_frame['radar_pts']
        pts_dynamic = radar_frame['dynamic_doppler'] > 0.3
        T_ego_local2global_k = radar_frame['T_ego_local2global']
        
        radar_pts = transform_points_to_current_frame(radar_pcl[:,:3], T_ego_local2global_k, T_ego_global2local_current)

        ax.scatter(radar_pts[:,0][pts_dynamic], radar_pts[:,1][pts_dynamic], color='blue', s=2) # dynamic points
        ax.scatter(radar_pts[:,0][~pts_dynamic], radar_pts[:,1][~pts_dynamic], color='orange', s=2) # dynamic points

    
    # draw a bounding boxes using current (last) frame targets
    for i in range(radar_frame['targets_bboxes'].shape[0]):
        cx, cy, _, ex, ey, _, yaw = radar_frame['targets_bboxes'][i]
        rect = Rectangle(
            xy=(cx-ex/2, cy-ey/2),
            width=ex,
            height=ey, 
            angle=yaw,
            rotation_point='center',
            edgecolor='red',
            facecolor='none',
            fill=False,
        )
        ax.add_patch(rect)

    ax.grid(True)
    ax.set_aspect('equal')
    plt.show()
    
def visualize_radar_pcl_aggregated_fixed(radar_dataset, start_idx, end_idx):
    '''
        Function aggregates radar frames from start_idx to end_idx.
    '''
    fig, ax = plt.subplots()
    
    T_ego_local2global_current = radar_dataset[end_idx-1]['T_ego_local2global']
    T_ego_global2local_current = np.linalg.inv(T_ego_local2global_current)

    # draw radar points for every aggregated frame
    for k in range(start_idx, end_idx):
        radar_frame = radar_dataset[k]
        radar_pcl = radar_frame['radar_pts']
        pts_dynamic = radar_frame['dynamic_doppler'] > 0.3
        
        T_ego_local2global_k = radar_frame['T_ego_local2global']
        radar_pts = transform_points_to_current_frame(radar_pcl[:,:3], T_ego_local2global_k, T_ego_global2local_current)

        ax.scatter(radar_pts[:,0][pts_dynamic], radar_pts[:,1][pts_dynamic], color='blue', s=15) # dynamic points
        ax.scatter(radar_pts[:,0][~pts_dynamic], radar_pts[:,1][~pts_dynamic], color='orange', s=15) # dynamic points
    
    
    # draw a bounding boxes using current (last) frame targets
    for i in range(radar_frame['targets_bboxes'].shape[0]):
        cx, cy, _, ex, ey, _, yaw = radar_frame['targets_bboxes'][i]
        rect = Rectangle(
            xy=(cx-ex/2, cy-ey/2),
            width=ex,
            height=ey, 
            angle=yaw,
            rotation_point='center',
            edgecolor='red',
            facecolor='none',
            fill=False,
        )
        ax.add_patch(rect)

    ax.grid(True)
    ax.set_aspect('equal')
    plt.show()