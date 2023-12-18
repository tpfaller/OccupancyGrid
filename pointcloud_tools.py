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


def compute_length_and_degree(pcd):
    pcd = pcd[:, :2]
    vector_length = np.linalg.norm(pcd, ord=2, axis=-1)
    angle = np.arctan2(pcd[:,1], pcd[:,0])
    degree = np.rad2deg(angle)

    # For debugging
    # table = np.concatenate([pcd, degree.reshape((-1, 1))], axis=-1).tolist()
    # pprint([f"{x[0]:<4.2f}  {x[1]:<4.2f}  {x[2]:<3.2f}" for x in table])
    return vector_length, degree


def count_points_per_cell(pcd: np.ndarray, angles: np.ndarray, circles: np.ndarray) -> np.ndarray:
        vector_lengths, pcd_angles = compute_length_and_degree(pcd)

        angle_correction = np.select([pcd_angles < 0], [360], default=0)
        pcd_angles += angle_correction
        
        angle_bins = np.digitize(pcd_angles, angles)
        radius_bins = np.digitize(vector_lengths, circles)

        occupancy_cells = np.zeros((circles.shape[0]+ 1, angles.shape[0] + 1))
        for radius_bin, angle_bin in zip(radius_bins, angle_bins):
            occupancy_cells[radius_bin, angle_bin] += 1
        return occupancy_cells



def get_rings(ego_center: float = 6.,
ring_width: float = 1.,
last_ring: float = 100,
growth_rate: float = 1.1
) -> np.ndarray:
    ring_size = ego_center
    rings = [ring_size]

    while ring_size < last_ring:
        ring_size += ring_width
        ring_width *= growth_rate
        rings.append(ring_size)
    return np.array(rings)
