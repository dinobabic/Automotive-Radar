import numpy as np
import pickle as pkl
import glob

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

        return radar_frame