import argparse
import os
from typing import Tuple, List, Dict

import cv2
import numpy as np
from nuscenes.nuscenes import NuScenes

from pointcloud_tools import get_calibrated_pointcloud, filter_pointcloud
from utils_occupancy_grid import get_rings, compute_length_and_degree, count_points_per_cell
from visualization import show_image, concatenate_all_images, draw_raw_pointcloud
from dempster_shafer import dempster_shafer_theory


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--viz_raw_pointcloud", action='store_true')
    parser.add_argument("--square_grid", action='store_true')
    parser.add_argument("--round_grid", action='store_true')
    parser.add_argument("--scene_number", type=int, choices=range(0, 10))
    parser.add_argument("--data_path", type=str, default="data/v1.0-mini")

    parser.add_argument("--camera_name", type=str, default="CAM_FRONT")
    parser.add_argument("--resolution", type=int, default=300)

    # Pointcloud filter parameter
    parser.add_argument("--min_height", type=float, default=0.3)
    parser.add_argument("--max_height", type=float, default=None)
    parser.add_argument("--min_x", type=float, default=1.5)
    parser.add_argument("--min_y", type=float, default=1.0)

    args = parser.parse_args()

    args.max_height = args.max_height if args.max_height else np.inf
    return args


def init_nuscene(data_path: str, scene_number: int) -> Tuple[NuScenes, dict]:
    nusc = NuScenes(version='v1.0-mini', dataroot=data_path, verbose=False)
    scene = nusc.scene[scene_number]
    sample = nusc.get('sample', scene['first_sample_token'])
    return nusc, sample


def get_camera_image(camera_name: str, nusc: NuScenes, sample: dict, resolution: int) -> np.ndarray:
    cam = nusc.get('sample_data', sample['data'][camera_name])
    file_path = os.path.join("data/v1.0-mini/", cam["filename"])
    image = cv2.imread(file_path)
    return cv2.resize(image, (resolution, resolution))


def get_radar_map(sensor_data: Dict[str, np.ndarray], resolution: int) -> np.ndarray:
    pointcloud = np.empty((0, 2))
    for sensor, sensor_pcd in sensor_data.items():
        if "RADAR" in sensor:
            pointcloud = np.vstack([pointcloud, sensor_pcd])

    discret_pcd = np.round(pointcloud).astype(np.int32)
    center = resolution // 2
    discret_pcd_centered = np.array([center, center]) - discret_pcd

    background = np.zeros((resolution, resolution), dtype=np.float32)
    for (x, y) in discret_pcd_centered:
            background[x, y] = 1.
    return background
    

def get_lidar_map(sensor_data: Dict[str, np.ndarray], resolution: int) -> np.ndarray:
    pcd = sensor_data["LIDAR_TOP"]

    discret_pcd = np.round(pcd).astype(np.int32)
    center = resolution // 2
    discret_pcd_centered = np.array([center, center]) - discret_pcd
    
    background = np.zeros((resolution, resolution), dtype=np.float32)
    for (x, y) in discret_pcd_centered:
            background[x, y] = 1.
    return background


def init_round_grid(resolution: int) -> Tuple[np.ndarray, np.ndarray]:
    angle_resolution = 2

    rings = get_rings(last_ring=resolution//2, growth_rate=1.0)
    angles = np.array([x for x in range(0, 360, angle_resolution)])
    return rings, angles
    

def discretize_radar(sensor_data: Dict[str, np.ndarray], rings: np.ndarray, angles: np.ndarray) -> np.ndarray:
    pointcloud = np.empty((0, 2), dtype=np.float32)
    for sensor, sensor_pcd in sensor_data.items():
        if "RADAR" in sensor:
            pointcloud = np.vstack([pointcloud, sensor_pcd])
    length, degree = compute_length_and_degree(pointcloud=pointcloud)
    radar_map = count_points_per_cell(
        vector_lengths=length,
        pcd_angles=degree,
        angles=angles,
        circles=rings
    )
    return radar_map


def discretize_lidar(sensor_data: Dict[str, np.ndarray], rings: np.ndarray, angles: np.ndarray) -> np.ndarray:
    pointcloud = sensor_data["LIDAR_TOP"]
    
    length, degree = compute_length_and_degree(pointcloud=pointcloud)
    radar_map = count_points_per_cell(
        vector_lengths=length,
        pcd_angles=degree,
        angles=angles,
        circles=rings
    )
    return radar_map


def main() -> None: 
    args = get_args()
    # Init Nuscene, Select Scene, get first sample
    nusc, sample = init_nuscene(data_path=args.data_path, scene_number=args.scene_number)
    sensor_list = ["LIDAR_TOP", "RADAR_FRONT", "RADAR_FRONT_LEFT", "RADAR_BACK_LEFT", "RADAR_BACK_RIGHT", "RADAR_FRONT_RIGHT"]
    
    while len(sample["next"]) > 0:
        images = []
        sensor_data = {}
        for sensor in sensor_list:
            # Get all sensor data and project pointclouds into ego pose
            pointcloud = get_calibrated_pointcloud(nusc=nusc, sample=sample, sensor_name=sensor)
            # Filter pointcloud
            pointcloud = filter_pointcloud(
                pcd=pointcloud,
                min_height=args.min_height,
                max_height=args.max_height,
                min_x=args.min_x,
                min_y=args.min_y,
                max_dist=(args.resolution //2) - 1
            )
            # Project pointcloud into birds eye view
            pointcloud = pointcloud[:, :2]
            sensor_data[sensor] = pointcloud

        if args.viz_raw_pointcloud:
            raw_pointcloud_image = draw_raw_pointcloud(sensor_data=sensor_data, resolution=args.resolution)
            images.append(raw_pointcloud_image)
        if args.square_grid:
            # discretize pointcloud into square grid
            lidar_map = get_lidar_map(sensor_data, resolution=args.resolution)
            radar_map = get_radar_map(sensor_data, resolution=args.resolution)
            # dempster shafer theory
            occupancy_probability = dempster_shafer_theory(
                 lidar_grid=lidar_map,
                 radar_grid=radar_map,
                 m1_theta=.5,
                 m2_theta=.7
            )
            probs = cv2.cvtColor(occupancy_probability, cv2.COLOR_GRAY2RGB)
            images.append(probs.astype(np.uint8))
            # edge detection
            laplacian = cv2.Laplacian(occupancy_probability,cv2.CV_32F)
            laplacian = cv2.cvtColor(laplacian, cv2.COLOR_GRAY2RGB)
            images.append(laplacian.astype(np.uint8))

        if args.round_grid:
            rings, degrees = init_round_grid(resolution=args.resolution)
            
            radar_map = discretize_radar(sensor_data=sensor_data, rings=rings, angles=degrees)
            # radar_map = cv2.cvtColor(radar_map.astype(np.float32) * 255, cv2.COLOR_GRAY2RGB)
            # radar_map = cv2.resize(radar_map, (args.resolution, args.resolution), interpolation=cv2.INTER_NEAREST)
            # images.append(radar_map.astype(np.uint8))

            lidar_map = discretize_lidar(sensor_data=sensor_data, rings=rings, angles=degrees)
            # lidar_map = cv2.cvtColor(lidar_map.astype(np.float32) * 255, cv2.COLOR_GRAY2RGB)
            # lidar_map = cv2.resize(lidar_map, (args.resolution, args.resolution), interpolation=cv2.INTER_NEAREST)
            # images.append(lidar_map.astype(np.uint8))

            # dempster shafer theory
            occupancy_probability = dempster_shafer_theory(
                 lidar_grid=lidar_map,
                 radar_grid=radar_map,
                 m1_theta=.5,
                 m2_theta=.7
            )
            probs = cv2.cvtColor(occupancy_probability, cv2.COLOR_GRAY2RGB)
            probs = cv2.resize(probs, (args.resolution, args.resolution), interpolation=cv2.INTER_NEAREST)
            images.append(probs.astype(np.uint8))
            # edge detection
            laplacian = cv2.Laplacian(occupancy_probability,cv2.CV_32F)
            laplacian = cv2.cvtColor(laplacian, cv2.COLOR_GRAY2RGB)
            laplacian = cv2.resize(laplacian, (args.resolution, args.resolution), interpolation=cv2.INTER_NEAREST)
            images.append(laplacian.astype(np.uint8))
        
        # Get camera image
        camera_image = get_camera_image(
            camera_name=args.camera_name,
            nusc=nusc,
            sample=sample,
            resolution=args.resolution
        )

        images.insert(0, camera_image)
        # concatenate all images
        frame = concatenate_all_images(images=images)
        # show result
        show_image(image=frame, time=0)

        # Get next sample
        sample = nusc.get('sample', sample["next"])

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
