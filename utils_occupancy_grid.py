from typing import Tuple
import cv2
import numpy as np


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


def compute_length_and_degree(pointcloud: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    pointcloud = pointcloud[:, :2]
    vector_length = np.linalg.norm(pointcloud, ord=2, axis=-1)
    angle = np.arctan2(pointcloud[:,1], pointcloud[:,0])
    degree = np.rad2deg(angle)
    return vector_length, degree


def count_points_per_cell(vector_lengths: np.ndarray, pcd_angles: np.ndarray, angles: np.ndarray, circles: np.ndarray) -> np.ndarray:
        angle_bins = np.digitize(pcd_angles, angles)
        radius_bins = np.digitize(vector_lengths, circles)        

        occupancy_cells = np.zeros((angles.shape[0] + 1, circles.shape[0]+ 1), dtype=np.float32)
        for radius_bin, angle_bin in zip(radius_bins, angle_bins):
            occupancy_cells[angle_bin, radius_bin] += 1

        return occupancy_cells


def polar_coordinates_to_cartesian_grid(polar_grid: np.ndarray, cartesian_resolution: int, rings: np.ndarray):
    cart = cv2.warpPolar(
        src=polar_grid, 
        dsize=(cartesian_resolution, cartesian_resolution),
        center=(cartesian_resolution//2, cartesian_resolution//2),
        maxRadius=rings[-1],
        # maxRadius=cartesian_resolution//2,
        flags=cv2.WARP_INVERSE_MAP)
    return cart
