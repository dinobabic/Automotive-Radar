import numpy as np
import pickle as pkl
import glob
import matplotlib.pyplot as plt

from utils import transform_points_to_current_frame, normalize_to_bev

import torch
from torch.utils.data import Dataset


'''
NOTES:
    - coordinate frame: y - front/back
                        x - right/left
                        z - up/down

    radar_frame - dictionary with keys:
            - radar_pts: (N, 8) - [x, y, z, azimuth, elevation, range, doppler, RCS], angles are in degrees
            - ego_vel: (3) - [vel_x, vel_y, vel_z]
            - timestamp: frame timestamp in seconds
            - targets_bboxes: (K,7): GT bboxes [cx, cy, cz, ex, ey, ez, yaw(deg)]
            - T_ego_local2global (4,4): convert from ego local coordinates to global coordinates
            - targets_vel (K,3): Target vehicles GT velocities
            - target_ids (K,): Target vehicles unique ID
            - target_is_visible (K,): Flag indicating if target vehicle is visible to sensor
'''

class RadarDataset(Dataset):
    def __init__(self, root_radar_dir, train=True):
        super().__init__()

        self.radar_files = glob.glob(f'{root_radar_dir}/train/*') if train else glob.glob(f'{root_radar_dir}/test/*')
        self.radar_files.sort()

    def __len__(self):
        return len(self.radar_files)
    
    def __getitem__(self, idx):
        with open(self.radar_files[idx], 'rb') as file:
            radar_frame = pkl.load(file)

        # ego vehicle's motion compensation
        ego_vel = radar_frame['ego_vel']
        dynamic_doppler = np.zeros(radar_frame['radar_pts'].shape[0], dtype=np.float32)
        for i, point in enumerate(radar_frame['radar_pts']):
            azimuth = point[3] # azimuth angle
            h = ego_vel[0] * np.sin(np.deg2rad(azimuth)) + ego_vel[1] * np.cos(np.deg2rad(azimuth)) # radial component of ego vehicle's velocity
            # remove ego vehicle's radial velocity component from point's doppler to get dynamic doppler
            dynamic_doppler[i] = point[6]+h
        
        radar_frame['dynamic_doppler'] = dynamic_doppler
        return radar_frame
    
class RadarBEVDataset(Dataset):
    def __init__(self, radar_dataset, aggregate=False, aggregation_time=0.5):
        super().__init__()

        self.radar_dataset = radar_dataset
        self.aggregate = aggregate
        self.aggregation_time = aggregation_time

    def __len__(self):
        return len(self.radar_dataset)
    
    def __getitem__(self, idx):
        radar_frame = self.radar_dataset[idx]

        start_idx = idx
        init_timestamp = radar_frame['timestamp']
        idx -= 1
        while idx >= 0:
            prev_timestamp = self.radar_dataset[idx]['timestamp']
            if init_timestamp - prev_timestamp < self.aggregation_time:
                idx -= 1
            else:
                break

        last_timestamp = self.radar_dataset[idx+1]['timestamp']

        T_ego_local2global_current = radar_frame['T_ego_local2global']
        T_ego_global2local_current = np.linalg.inv(T_ego_local2global_current)

        # store target labels
        target_bbox = np.array(radar_frame['targets_bboxes']) 

        # add timestamps to radar points
        radar_pcl = radar_frame['radar_pts'].copy()
        radar_pcl = np.hstack([radar_pcl, np.full((radar_pcl.shape[0], 1), init_timestamp)])
        aggregated_radar_pcl = np.array(radar_pcl)
        
        if self.aggregate:
            for k in range(idx+1, start_idx):
                radar_frame_k = self.radar_dataset[k]
                radar_pcl = radar_frame_k['radar_pts'].copy()
                T_ego_local2global_k = radar_frame_k['T_ego_local2global']

                # transfrom radar points from frame k to the current frame 
                radar_pts_transformed = transform_points_to_current_frame(radar_pcl[:,:3], T_ego_local2global_k, T_ego_global2local_current)
                radar_pcl[:, :3] = radar_pts_transformed

                # add timestamps to radar points
                radar_pcl = np.hstack([radar_pcl, np.full((radar_pcl.shape[0], 1), radar_frame_k['timestamp'])])
                aggregated_radar_pcl = np.vstack([aggregated_radar_pcl, radar_pcl])

        # filter out points with low RCS (less than 40dB)
        filter_mask = aggregated_radar_pcl[:, 7] > 40
        aggregated_radar_pcl = aggregated_radar_pcl[filter_mask]

        # create BEV
        # ego car is at (0, 0) in the BEV, x-axis points to the right, y-axis points forward
        # BEV covers 100m in front of the ego car, and 50m to the left and right of the ego car, with a resolution of 0.25m
        # there is no need to create BEV behind the car as we only have one radar looking forward
        bev_width = 400 # number of pixels in x direction (100m / 0.25m)
        bev_height = 400 # number of pixels in y direction (100m / 0.25m)

        # 5 channels: [doppler, elevation, azimuth, RCS, relative detection timestamp] - mean of each cell and normalized to [0, 1]
        bev = np.zeros((5, bev_height, bev_width), dtype=np.float32) 
        cell_count = np.zeros((bev_height, bev_width), dtype=np.float32)

        # filter points that are outside the BEV range
        x_min, x_max = -50, 50
        y_min, y_max = 0, 100
        filter_mask = (aggregated_radar_pcl[:, 0] >= x_min) & (aggregated_radar_pcl[:, 0] < x_max) & \
                      (aggregated_radar_pcl[:, 1] >= y_min) & (aggregated_radar_pcl[:, 1] < y_max)
        aggregated_radar_pcl = aggregated_radar_pcl[filter_mask]

        # normalization
        aggregated_radar_pcl[:, 0] = (aggregated_radar_pcl[:, 0] - x_min) / (x_max - x_min) * (bev_width - 1)
        aggregated_radar_pcl[:, 1] = (aggregated_radar_pcl[:, 1] - y_min) / (y_max - y_min) * (bev_height - 1)
        aggregated_radar_pcl[:, 3] = np.clip((aggregated_radar_pcl[:, 3] + 180) / 360, 0, 1)
        aggregated_radar_pcl[:, 4] = np.clip((aggregated_radar_pcl[:, 4] + 90) / 180, 0, 1) 
        aggregated_radar_pcl[:, 6] = np.clip((aggregated_radar_pcl[:, 6] + 100) / 200, 0, 1)
        aggregated_radar_pcl[:, 7] = np.clip(aggregated_radar_pcl[:, 7] / 100, 0, 1) 
        aggregated_radar_pcl[:, 8] = (init_timestamp - aggregated_radar_pcl[:, 8]) / (init_timestamp - last_timestamp) 

        # aggregate points in each cell 
        for i in range(aggregated_radar_pcl.shape[0]):
            x_idx = int(aggregated_radar_pcl[i, 0])
            y_idx = int(aggregated_radar_pcl[i, 1])
            # flip y-axis for BEV visualization - top row corresponds with high y values
            y_idx = bev_height - 1 - y_idx
            bev[:, y_idx, x_idx] += aggregated_radar_pcl[i, [6, 4, 3, 7, 8]] # [doppler, elevation, azimuth, RCS, relative detection timestamp]
            cell_count[y_idx, x_idx] += 1

        # average the aggregated points
        mask = cell_count > 0
        bev[:, mask] /= cell_count[mask]

        # project target bboxes to BEV
        #target_bbox_bev = np.zeros((target_bbox.shape[0], 5), dtype=np.float32) # [cx, cy, ex, ey, yaw]
        target_bbox_bev = []
        for i, bbox in enumerate(target_bbox):
            cx, cy, cz, ex, ey, ez, yaw = bbox

            # filter out targets further than 70m
            if cy > 70:
                continue
            
            cx_bev_norm = (cx - x_min) / (x_max - x_min) * (bev_width - 1)
            cy_bev_norm = (cy - y_min) / (y_max - y_min) * (bev_height - 1)
            cy_bev = bev_height - 1 - cy_bev_norm
            cx_bev = cx_bev_norm
            
            ex_bev = ex / (x_max - x_min) * (bev_width - 1)
            ey_bev = ey / (y_max - y_min) * (bev_height - 1)

            target_bbox_bev.append([cx_bev, cy_bev, ex_bev, ey_bev, -yaw])

        target_bbox_bev = np.array(target_bbox_bev)

        # filter target bboxes
        valid_target_bboxes = []
        for i, bbox in enumerate(target_bbox_bev):
            cx, cy, ex, ey, yaw = bbox

            # if bounding box is outside of the BEV range
            if cx < 0 or cx >= bev_width or cy < 0 or cy >= bev_height:
                continue
            
            # filter out targets that have less than 4 radar points in their bounding box 
            x_min_bbox = int(max(cx - ex/2, 0))
            x_max_bbox = int(min(cx + ex/2, bev_width - 1))
            y_min_bbox = int(max(cy - ey/2, 0))
            y_max_bbox = int(min(cy + ey/2, bev_height - 1))

            ys, xs = np.meshgrid(
                np.arange(y_min_bbox, y_max_bbox + 1),
                np.arange(x_min_bbox, x_max_bbox + 1),
                indexing='ij'
            )

            xs = xs.reshape(-1)
            ys = ys.reshape(-1) 

            rect = plt.Rectangle(
                (cx - ex/2, cy - ey/2), 
                ex, 
                ey, 
                angle=yaw, 
                rotation_point='center',
                edgecolor='red', 
                facecolor='none'
            )

            poly = rect.get_path().transformed(rect.get_transform())
            pts = np.column_stack([xs, ys])
            mask = poly.contains_points(pts)
            
            valid_xs = xs[mask].astype(int)
            valid_ys = ys[mask].astype(int)
            count = np.sum(cell_count[valid_ys, valid_xs])
            
            if count >= 4:
                valid_target_bboxes.append([cx, cy, ex, ey, np.sin(np.deg2rad(yaw)), np.cos(np.deg2rad(yaw))])    

        
        valid_target_bboxes = np.array(valid_target_bboxes) 

        # visualization
        # fig, ax = plt.subplots()
        # plt.imshow(bev[0], cmap='hot', interpolation='nearest')
        # for i in range(valid_target_bboxes.shape[0]):
        #     cx, cy, ex, ey, yaw = valid_target_bboxes[i]

        #     rect = plt.Rectangle(
        #         (cx - ex/2, cy - ey/2), 
        #         ex, 
        #         ey, 
        #         angle=yaw, 
        #         rotation_point='center',
        #         edgecolor='red', 
        #         facecolor='none'
        #     )

        #     ax.add_patch(rect)

        # plt.show()

        return torch.tensor(bev, dtype=torch.float32), torch.tensor(valid_target_bboxes, dtype=torch.float32)


        