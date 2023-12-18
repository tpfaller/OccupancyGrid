import argparse
import os
from typing import Tuple, List

import cv2
import numpy as np
from nuscenes.nuscenes import NuScenes

from pointcloud_tools import get_calibrated_pointcloud, filter_pointcloud
from visualization import show_image, concatenate_all_images, draw_raw_pointcloud


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


def get_radar_map(sensor_data: dict, resolution: int) -> np.ndarray:
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
    


def get_lidar_map(sensor_data: dict, resolution: int) -> np.ndarray:
    pcd = sensor_data["LIDAR_TOP"]

    discret_pcd = np.round(pcd).astype(np.int32)
    center = resolution // 2
    discret_pcd_centered = np.array([center, center]) - discret_pcd
    
    background = np.zeros((resolution, resolution), dtype=np.float32)
    for (x, y) in discret_pcd_centered:
            background[x, y] = 1.
    return background


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
            square_grid = None
            # discretize pointcloud into square grid
            lidar_map = get_lidar_map(sensor_data, resolution=args.resolution)
            radar_map = get_radar_map(sensor_data, resolution=args.resolution)
            # dempster shafer theory
            # edge detection

            # gray = cv2.cvtColor(square_grid, cv2.COLOR_BGR2GRAY)
            # laplacian = cv2.Laplacian(gray,cv2.CV_64F)
            # images.append(laplacian)

        if args.round_grid:
            round_grid = None
            # discretize pointcloud into square grid
            # dempster shafer theory
            # edge detection
            gray = cv2.cvtColor(round_grid, cv2.COLOR_BGR2GRAY)
            laplacian = cv2.Laplacian(gray,cv2.CV_64F)
            images.append(laplacian)
        
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
        show_image(image=frame)

        # Get next sample
        sample = nusc.get('sample', sample["next"])

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
