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

def visualize_lidar(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    if len(points) > 0:
        z_min, z_max = points[:, 2].min(), points[:, 2].max()
        colors = np.zeros((len(points), 3))
        if z_max > z_min:
            normalized_z = (points[:, 2] - z_min) / (z_max - z_min)
            colors[:, 0] = normalized_z  
            colors[:, 2] = 1 - normalized_z
        else:
            colors[:, 2] = 1  
        pcd.colors = o3d.utility.Vector3dVector(colors)
    
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5)
    o3d.visualization.draw_geometries([pcd, mesh_frame])

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

idx = 0
img = cv.imread(image_files[idx], cv.IMREAD_COLOR)
labels = read_labels(label_files[idx])
calib = read_calibration(calibration_files[idx])
lidar_points = read_lidar_pcl(lidar_files[idx])

img = draw_labels_on_image(img, labels)
visualize_lidar(lidar_points)

cv.imshow('Image', img)
cv.waitKey(0)