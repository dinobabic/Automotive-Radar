import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import glob
import open3d as o3d
 
class Label:
    def __init__(self, obj_class, truncation, occlusion, alpha, 
                bbox_left, bbox_top, bbox_right, bbox_bottom,
                height, width, length,
                x, y, z,
                rotation_y,
                id):
        self.obj_class = obj_class
        self.truncation = truncation   
        self.occlusion = occlusion
        self.alpha = alpha
        self.bbox_left = bbox_left
        self.bbox_top = bbox_top
        self.bbox_right = bbox_right
        self.bbox_bottom = bbox_bottom
        self.height = height
        self.width = width
        self.length = length
        self.x = x
        self.y = y
        self.z = z
        self.rotation_y = rotation_y
        self.id = id

def read_calibration(calib_file):
    calib = {}
    with open(calib_file, 'r') as f:
        for line in f:
            key, value = line.strip().split(':', 1)
            calib[key] = np.array([float(x) for x in value.split()])
            if key.startswith('P') or key.startswith('T'):
                calib[key] = calib[key].reshape(3, 4)
            elif key.startswith('R'):
                calib[key] = calib[key].reshape(3, 3)
    return calib

def read_labels(label_file):
    labels = []
    with open(label_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            label = Label(
                obj_class=parts[0],
                truncation=float(parts[1]),
                occlusion=int(parts[2]),
                alpha=float(parts[3]),
                bbox_left=float(parts[4]),
                bbox_top=float(parts[5]),
                bbox_right=float(parts[6]),
                bbox_bottom=float(parts[7]),
                height=float(parts[8]),
                width=float(parts[9]),
                length=float(parts[10]),
                x=float(parts[11]),
                y=float(parts[12]),
                z=float(parts[13]),
                rotation_y=float(parts[14]),
                id=parts[15]
            )
            labels.append(label)
    return labels

def draw_labels_on_image(img, labels: list[Label]):
    for label in labels:
        if label.obj_class in ['Car', 'Pedestrian', 'Cyclist']:
            cv.rectangle(img, (int(label.bbox_left), int(label.bbox_top)), 
                            (int(label.bbox_right), int(label.bbox_bottom)), 
                            (0, 255, 0), 2)
            cv.putText(img, label.obj_class, (int(label.bbox_left), int(label.bbox_top) - 10), 
                        cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
    return img

def read_lidar_pcl(lidar_file):
    points = np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)
    valid_mask = ~np.isnan(points[:, 0])
    return points[valid_mask, :3]  

def read_radar_pcl(radar_file):
    points = np.fromfile(radar_file, dtype=np.float32).reshape(-1, 5)
    valid_mask = ~np.isnan(points[:, 0])
    return points  

def visualize_pcl(points, labels: list[Label], Tr_velo_to_cam, Tr_radar_to_velo=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    colors = np.zeros(points.shape, dtype=np.float32)
    colors[:, 0] = 1.0
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Collect all geometries to visualize
    geometries = [pcd]
    
    # Compute inverse transformation (camera to velodyne)
    if Tr_radar_to_velo is not None:
        Tr_velo_to_cam = np.vstack([Tr_velo_to_cam, np.array([0, 0, 0, 1])])
        Tr_radar_to_velo = np.vstack([Tr_radar_to_velo, np.array([0, 0, 0, 1])])
        Tr_velo_to_cam = Tr_velo_to_cam @ Tr_radar_to_velo
    
    R_velo_to_cam = Tr_velo_to_cam[:3, :3]
    t_velo_to_cam = Tr_velo_to_cam[:3, 3]

    Tr_cam_to_velo = np.eye(4)
    Tr_cam_to_velo[:3, :3] = R_velo_to_cam.T
    Tr_cam_to_velo[:3, 3] = -R_velo_to_cam.T @ t_velo_to_cam
    
    # Draw bounding boxes for each label
    for label in labels:
        x_corners = [label.length/2, label.length/2, -label.length/2, -label.length/2,
                        label.length/2, label.length/2, -label.length/2, -label.length/2]
        y_corners = [0, 0, 0, 0, -label.height, -label.height, -label.height, -label.height]
        z_corners = [label.width/2, -label.width/2, -label.width/2, label.width/2,
                        label.width/2, -label.width/2, -label.width/2, label.width/2]
        corners_3d = np.array([x_corners, y_corners, z_corners])
        R = np.array([[np.cos(label.rotation_y), 0, np.sin(label.rotation_y)],
                        [0, 1, 0],
                        [-np.sin(label.rotation_y), 0, np.cos(label.rotation_y)]])
        corners_3d = R @ corners_3d
        corners_3d[0] += label.x
        corners_3d[1] += label.y
        corners_3d[2] += label.z
        
        # Transform from camera coordinates to velodyne coordinates
        # Add homogeneous coordinate for transformation
        ones = np.ones((1, corners_3d.shape[1]))
        corners_3d_homo = np.vstack([corners_3d, ones])
        corners_3d_velo_homo = Tr_cam_to_velo @ corners_3d_homo
        corners_3d = corners_3d_velo_homo[:3, :]
        
        # Create lines for the bounding box
        lines = [[0,1], [1,2], [2,3], [3,0], 
                    [4,5], [5,6], [6,7], [7,4], 
                    [0,4], [1,5], [2,6], [3,7]]
        
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(corners_3d.T)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector([[0, 1, 0] for _ in lines])
        geometries.append(line_set)
    
    # Add coordinate frame for reference
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5)
    geometries.append(mesh_frame)
    
    # Visualize all geometries together with custom camera view
    # Camera looking down positive Z-axis from the origin
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='LiDAR Point Cloud with Labels')
    
    for geom in geometries:
        vis.add_geometry(geom)
    
    # Set camera parameters
    ctr = vis.get_view_control()
    # Set camera to look down the positive X-axis (forward direction)
    # front: direction from camera to lookat point
    # lookat: point the camera is looking at (origin)
    # up: which direction is "up"
    ctr.set_front([-0.1, 0, 0])      # Look in positive X direction (forward)
    ctr.set_lookat([0, 0, 0])     # Look at origin
    ctr.set_up([0, 0, 1])         # Z-axis points up
    ctr.set_zoom(0.5)             # Adjust zoom level
    
    vis.run()
    vis.destroy_window()


# possible subfolders:
    # - calib
    # - image_2
    # - label_2
    # - radar
    # - velodyne
root_dir = '/Volumes/T7/V2X-Radar-V/training'

image_files = glob.glob(f'{root_dir}/image_2/*')
image_files.sort()

label_files = glob.glob(f'{root_dir}/label_2/*')
label_files.sort()

calibration_files = glob.glob(f'{root_dir}/calib/*')   
calibration_files.sort()

lidar_files = glob.glob(f'{root_dir}/velodyne/*')  
lidar_files.sort()

radar_files = glob.glob(f'{root_dir}/radar/*')  
radar_files.sort()

idx = 0
img = cv.imread(image_files[idx], cv.IMREAD_COLOR)
labels = read_labels(label_files[idx])
calib = read_calibration(calibration_files[idx])
lidar_points = read_lidar_pcl(lidar_files[idx])
radar_points = read_radar_pcl(radar_files[idx])

img = draw_labels_on_image(img, labels)
visualize_pcl(lidar_points, labels, calib['Tr_velo_to_cam'])
visualize_pcl(radar_points[:, :3], labels, calib['Tr_velo_to_cam'], calib['Tr_radar_to_velo'])

cv.imshow('Image', img)
cv.waitKey(0)