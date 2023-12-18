from typing import Tuple
import os

import numpy as np 
import cv2
from nuscenes.nuscenes import NuScenes
from pointcloud_tools import get_calibrated_pointcloud, filter_pointcloud

def show_edges(grid: np.ndarray, time: int = 0) -> None:
    gray = cv2.cvtColor(grid, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray,cv2.CV_64F)
    # cv2.namedWindow("Image Front", cv2.WINDOW_NORMAL) 
    # cv2.resizeWindow("Image Front", 1280, 1280) 
    cv2.imshow("Image Front", laplacian)
    cv2.waitKey(time)

def show_grid_and_image(nusc: NuScenes, sample: dict, grid: np.ndarray, time: int = 0) -> None:
    cam_front = nusc.get('sample_data', sample['data']['CAM_FRONT'])
    file_path = os.path.join("data/v1.0-mini/", cam_front["filename"])
    image_front = cv2.imread(file_path)
    image_front = cv2.resize(image_front, grid.shape[:2])

    # cv2.namedWindow("Image Front", cv2.WINDOW_NORMAL) 
    # cv2.resizeWindow("Image Front", 1280, 1280) 
    cv2.imshow("Image Front", np.concatenate([grid, image_front], axis=1))
    cv2.waitKey(time)

def init_scene(data_path: str, scene_number: int):
    assert scene_number <10, print("Choose a number between 0 and 10.")
    nusc = NuScenes(version='v1.0-mini', dataroot=data_path, verbose=False)
    my_scene = nusc.scene[scene_number]
    my_sample = nusc.get('sample', my_scene['first_sample_token'])
    return nusc, my_sample

def init_grid(length: int):
    grid = np.zeros((length, length, 3), dtype=np.uint8)
    center = (grid.shape[0]//2, grid.shape[1]//2)
    grid[center] = (0, 255, 0)
    return grid, center

def discretize_and_center_pcd(pcd: np.ndarray, center: Tuple[int, int], scaling_factor: float) -> np.ndarray:
    pcd *= scaling_factor
    discret_pcd = np.round(pcd[:,:2]).astype(np.int32)
    discret_pcd_centered = np.array(center) - discret_pcd
    return discret_pcd_centered

def draw_sensor_data_into_grid(nusc: NuScenes, sample: dict, sensor: str, grid: np.ndarray, center: Tuple[int, int], scaling_factor: float):
    color = (0, 0, 255) if "RADAR" in sensor else (255, 0, 0)
    pcd_ego = get_calibrated_pointcloud(nusc, sample, sensor_name=sensor)
    pcd_ego = filter_pointcloud(pcd_ego, 
                                min_height=.3,
                                max_height=2.5,
                                min_x=1.5,
                                min_y=1.0,  
                                max_dist = int(center[0] / scaling_factor) -1
                                )
    discret_pcd_centered = discretize_and_center_pcd(pcd_ego, center=center, scaling_factor=scaling_factor)
    for (x, y) in discret_pcd_centered:
        grid[x, y, :] = color

def main():
    data_path = "data/v1.0-mini"
    nusc, sample = init_scene(data_path=data_path, scene_number=2)
    length = 300
    scaling_factor = length / 300
    grid, center = init_grid(length=length)

    while len(sample["next"]) > 0:
        grid, center = init_grid(length=length)
        # Draw Pointclouds into grid
        sensor_list = ["LIDAR_TOP", "RADAR_FRONT", "RADAR_FRONT_LEFT", "RADAR_BACK_LEFT", "RADAR_BACK_RIGHT", "RADAR_FRONT_RIGHT"]
        # sensor_list = ["RADAR_FRONT"]
        for sensor in sensor_list:
            draw_sensor_data_into_grid(
                nusc=nusc,
                sample=sample,
                sensor=sensor,
                grid=grid,
                center=center,
                scaling_factor=scaling_factor
            )
        # show_edges(grid=grid, time=0)
        show_grid_and_image(nusc, sample, grid=grid, time=0)
        sample = nusc.get('sample', sample["next"])

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
