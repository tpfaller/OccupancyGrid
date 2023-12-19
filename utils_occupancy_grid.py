from typing import Tuple

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

    # For debugging
    # table = np.concatenate([pcd, degree.reshape((-1, 1))], axis=-1).tolist()
    # pprint([f"{x[0]:<4.2f}  {x[1]:<4.2f}  {x[2]:<3.2f}" for x in table])
    return vector_length, degree


def count_points_per_cell(vector_lengths: np.ndarray, pcd_angles: np.ndarray, angles: np.ndarray, circles: np.ndarray) -> np.ndarray:
        # vector_lengths, pcd_angles = compute_length_and_degree(pcd)
        angle_correction = np.select([pcd_angles < 0], [360], default=0)
        pcd_angles += angle_correction
        
        angle_bins = np.digitize(pcd_angles, angles)
        radius_bins = np.digitize(vector_lengths, circles)

        occupancy_cells = np.zeros((circles.shape[0]+ 1, angles.shape[0] + 1), dtype=np.float32)
        for radius_bin, angle_bin in zip(radius_bins, angle_bins):
            occupancy_cells[radius_bin, angle_bin] += 1
        return occupancy_cells
