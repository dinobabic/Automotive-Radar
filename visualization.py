import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def visualize_radar_pcl(radar_frame: dict):
    radar_pcl = radar_frame['radar_pts']
    ego_vel = radar_frame['ego_vel']
    pts_dynamic = np.zeros(radar_pcl.shape[0], dtype=np.bool)

    for i, point in enumerate(radar_pcl):
        azimuth = point[3] # azimuth angle
        h = ego_vel[0] * np.sin(np.deg2rad(azimuth)) + ego_vel[1] * np.cos(np.deg2rad(azimuth)) # radial component of ego vehicle's velocity
        pts_dynamic[i] = np.abs(point[6]+h) > 0.3

    fig, ax = plt.subplots()
    
    ax.scatter(radar_pcl[:,0][pts_dynamic], radar_pcl[:,1][pts_dynamic], color='blue', s=15) # dynamic points
    ax.scatter(radar_pcl[:,0][~pts_dynamic], radar_pcl[:,1][~pts_dynamic], color='orange', s=15) # dynamic points
    
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

def visualize_radar_pcl_aggregated(radar_dataset, start_idx):
    '''
        Function aggregates radar frames inside a 0.5 seconds window and visualizes 
        radar point cloud together with the labels from the current frame (latest aggregated frame).
    '''
    init_timestamp = radar_dataset[start_idx]['timestamp']
    idx = start_idx + 1
    while (True):
        next_timestamp = radar_dataset[idx]['timestamp']
        print(f'init timestamp: {init_timestamp}, next timestamp: {next_timestamp}')
        if next_timestamp - init_timestamp < 0.5:
            idx += 1
        else:
            break
        
    fig, ax = plt.subplots()

    # draw radar points for every aggregated frame
    for k in range(start_idx, idx):
        radar_frame = radar_dataset[k]
        radar_pcl = radar_frame['radar_pts']
        ego_vel = radar_frame['ego_vel']
        pts_dynamic = np.zeros(radar_pcl.shape[0], dtype=np.bool)

        for i, point in enumerate(radar_pcl):
            azimuth = point[3] # azimuth angle
            h = ego_vel[0] * np.sin(np.deg2rad(azimuth)) + ego_vel[1] * np.cos(np.deg2rad(azimuth)) # radial component of ego vehicle's velocity
            pts_dynamic[i] = np.abs(point[6]+h) > 0.3

        ax.scatter(radar_pcl[:,0][pts_dynamic], radar_pcl[:,1][pts_dynamic], color='blue', s=15) # dynamic points
        ax.scatter(radar_pcl[:,0][~pts_dynamic], radar_pcl[:,1][~pts_dynamic], color='orange', s=15) # dynamic points
    
    
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