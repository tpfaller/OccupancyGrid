import numpy as np
import open3d as o3d

import os
from typing import Tuple
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import transform_matrix
from pyquaternion import Quaternion


def read_pcd_file(file_path):
    """
    Reads in a PCD file and returns a numpy array of the point cloud data.
    """
    pcd = o3d.io.read_point_cloud(file_path)
    out_arr = np.asarray(pcd.points)  
    return out_arr


def read_bin_file(file_path):
    scan = np.fromfile(file_path, dtype=np.float32)
    points = scan.reshape((-1, 5))[:, :4]
    return points


def get_calibrated_pointcloud(nusc: NuScenes, sample: dict, sensor_name: str) -> np.ndarray:
    # Get the sensor and calibration data
    sensor = nusc.get('sample_data', sample['data'][sensor_name])
    calibration_data = nusc.get("calibrated_sensor", sensor['calibrated_sensor_token'])

    # Get the transformation matrix via the saved calibration files
    P = transform_matrix(
                translation=np.array(calibration_data["translation"]),
                rotation=Quaternion(calibration_data["rotation"])
            )
    # Lidar pointcloud is saved as bin file
    if sensor['filename'][-3:] == "bin":
        pointcloud = read_bin_file(os.path.join("data/v1.0-mini", sensor['filename']))

        # Filter by reflection
        pointcloud = pointcloud[np.where((pointcloud[:, 3] > 5) & (pointcloud[:, 3] < 40))]
        # Discard reflection for projection
        pointcloud = pointcloud[:,:3]

    # Radar pointcloud is saved as pcd file
    elif sensor['filename'][-3:] == "pcd":
        pointcloud = read_pcd_file(os.path.join("data/v1.0-mini", sensor['filename']))

    # Create a homogenous 3d vector
    pointcloud = np.hstack([pointcloud, np.ones(pointcloud.shape[0]).reshape((-1, 1))])

    # Transform the pointcloud into the ego pose
    ego_pose = pointcloud @ P.T
    return ego_pose


def filter_pointcloud(pcd: np.ndarray, min_height: float = 0.3, max_height: float= np.inf, min_x: float = 1., min_y: float = 4., max_dist: float=150) -> np.ndarray:
    pcd = pcd[np.where((pcd[:, 2] > min_height) & (pcd[:, 2] < max_height))]
    pcd = pcd[np.where((np.abs(pcd[:, 0]) > min_x) | (np.abs(pcd[:, 1]) > min_y))]
    pcd = pcd[np.where(np.abs(pcd[:, 0]) < max_dist)]
    pcd = pcd[np.where(np.abs(pcd[:, 1]) < max_dist)]
    return pcd
