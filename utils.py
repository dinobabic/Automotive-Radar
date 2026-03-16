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