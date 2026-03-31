import numpy as np

def transform_points_to_current_frame(points, T_local2global_k, T_global2local_current):
    '''
        Transform points from the local coordinate frame of radar frame -k (current frame has idx 0)
        to the local coordinate frame of current radar frame (idx 0).    
    '''
    
    # convert points to homogeneous coordinates
    points_hom = np.hstack([points, np.ones((points.shape[0], 1))])
    
    # compute the transformation from radar frame -k to current radar frame
    T_k2current = T_global2local_current @ T_local2global_k
    
    # transform points to current radar frame
    points_current_hom = (T_k2current @ points_hom.T).T
    
    return points_current_hom[:, :3]


def normalize_to_bev(xy_points, x_min, x_max, y_min, y_max, bev_width, bev_height, flip_y_axis=True):
    '''
        Normalize 2D points (in physical coordinates) to BEV image coordinates.
        
        Args:
            xy_points: (N, 2) array of [x, y] points in physical coordinates
            x_min, x_max: x-axis range in physical coordinates
            y_min, y_max: y-axis range in physical coordinates
            bev_width, bev_height: BEV image dimensions in pixels
            flip_y_axis: whether to flip y-axis (for top-down visualization where row 0 is at top)
        
        Returns:
            normalized_points: (N, 2) array of [x_bev, y_bev] normalized to [0, bev_width-1] and [0, bev_height-1]
    '''
    normalized = xy_points.copy().astype(np.float32)
    
    # normalize x and y to [0, max-1] range
    normalized[:, 0] = (normalized[:, 0] - x_min) / (x_max - x_min) * (bev_width - 1)
    normalized[:, 1] = (normalized[:, 1] - y_min) / (y_max - y_min) * (bev_height - 1)
    
    # flip y-axis if requested (convert from coordinate system where y increases forward
    # to image coordinates where y increases downward)
    if flip_y_axis:
        normalized[:, 1] = bev_height - 1 - normalized[:, 1]
    
    return normalized