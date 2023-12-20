from typing import List, Tuple

import numpy as np
import cv2

from nuscenes import NuScenes

def discretize_and_center_pcd(pcd: np.ndarray, center: int, scaling_factor: float) -> np.ndarray:
    pcd *= scaling_factor
    discret_pcd = np.round(pcd[:,:2]).astype(np.int32)
    discret_pcd_centered = np.array([center, center]) - discret_pcd
    return discret_pcd_centered


def draw_raw_pointcloud(sensor_data: dict, resolution: int) -> np.ndarray:
    background = np.zeros((resolution, resolution, 3), dtype=np.uint8)
    for sensor, pcd in sensor_data.items():
        color = (0, 0, 255) if "RADAR" in sensor else (255, 0, 0)
        discret_pcd_centered = discretize_and_center_pcd(pcd, center=resolution // 2, scaling_factor=1.0)
        for (x, y) in discret_pcd_centered:
            background[x, y, :] = color

    cv2.circle(background, (resolution//2, resolution//2), 3,(0, 255, 0), -1)
    return background

def concatenate_all_images(images: List[np.ndarray]) -> np.ndarray:
    return np.concatenate(images, axis=1)


def show_image(image: np.ndarray, time: int=500) -> None:
    cv2.namedWindow("Stream occupancy grid", cv2.WINDOW_NORMAL) 
    cv2.resizeWindow("Stream occupancy grid", 1280, 1280) 
    cv2.imshow("Stream occupancy grid", image)
    cv2.waitKey(time)
