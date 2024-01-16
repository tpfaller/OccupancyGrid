import argparse
import os
from typing import Tuple, Dict

import cv2
import numpy as np
from nuscenes.nuscenes import NuScenes

from pointcloud_tools import get_calibrated_pointcloud, filter_pointcloud
from utils_occupancy_grid import get_rings, compute_length_and_degree, count_points_per_cell, polar_coordinates_to_cartesian_grid
from visualization import show_image, concatenate_all_images, draw_raw_pointcloud
from dempster_shafer import dempster_shafer_fusion


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--viz_raw_pointcloud", action='store_true')
    parser.add_argument("--square_grid", action='store_true')
    parser.add_argument("--round_grid", action='store_true')
    parser.add_argument("--scene_number", type=int, choices=range(0, 10))
    parser.add_argument("--data_path", type=str, default="data/v1.0-mini")

    parser.add_argument("--camera_name", type=str, default="CAM_FRONT")
    parser.add_argument("--resolution", type=int, default=300)

    # Round grid parameter
    parser.add_argument("--angle_resolution", type=int, default=2)
    parser.add_argument("--ring_width", type=float, default=1.0)
    parser.add_argument("--growth_rate", type=float, default=1.0)

    # Pointcloud filter parameter
    parser.add_argument("--min_height", type=float, default=0.3)
    parser.add_argument("--max_height", type=float, default=None)
    parser.add_argument("--min_x", type=float, default=1.5)
    parser.add_argument("--min_y", type=float, default=1.0)

    # Dempster-Shafer parameter
    parser.add_argument("--lidar_occupied", type=float)
    parser.add_argument("--lidar_unoccupied", type=float)
    parser.add_argument("--lidar_unsure", type=float)
    parser.add_argument("--radar_occupied", type=float)
    parser.add_argument("--radar_unoccupied", type=float)
    parser.add_argument("--radar_unsure", type=float)

    parser.add_argument("--time", type=int, default=0, 
                        help="Time for one frame in milliseconds. 0 is wating for pressing arbitrary key.")
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


def init_round_grid(cartesian_resolution: int, angle_resolution: int, ring_width, growth_rate: float) -> Tuple[np.ndarray, np.ndarray]:
    rings = get_rings(
        ego_center=0,
        ring_width=ring_width,
        growth_rate=growth_rate,
        last_ring=cartesian_resolution//2)
    angles = np.array([x for x in range(-180, 180, angle_resolution)])
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
    radar_map = np.select([radar_map > .0], [1.], default=0.)
    return radar_map


def discretize_lidar(sensor_data: Dict[str, np.ndarray], rings: np.ndarray, angles: np.ndarray) -> np.ndarray:
    pointcloud = sensor_data["LIDAR_TOP"]
    
    length, degree = compute_length_and_degree(pointcloud=pointcloud)
    lidar_map = count_points_per_cell(
        vector_lengths=length,
        pcd_angles=degree,
        angles=angles,
        circles=rings
    )

    lidar_map = np.select([lidar_map > .0], [1.], default=0.)
    return lidar_map


def main() -> None: 
    args = get_args()
    # Init Nuscene, Select Scene, get first sample
    nusc, sample = init_nuscene(data_path=args.data_path, scene_number=args.scene_number)
    sensor_list = ["LIDAR_TOP", "RADAR_FRONT", "RADAR_FRONT_LEFT", "RADAR_BACK_LEFT", "RADAR_BACK_RIGHT", "RADAR_FRONT_RIGHT"]
    dempster_shafer_parameter = {
        "lidar_occupied": args.lidar_occupied,
        "lidar_unoccupied": args.lidar_unoccupied,
        "lidar_unsure": args.lidar_unsure,
        "radar_occupied": args.radar_occupied,
        "radar_unoccupied": args.radar_unoccupied,
        "radar_unsure": args.radar_unsure,
    }
    
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
            occupancy_probability = dempster_shafer_fusion(
                lidar_grid=lidar_map,
                radar_grid=radar_map,
                weights=dempster_shafer_parameter
            )

            occupancy_color_coded = np.expand_dims(occupancy_probability * 255, axis=-1)
            occupancy_color_coded = np.concatenate([
                np.zeros_like(occupancy_color_coded),
                occupancy_color_coded,
                np.zeros_like(occupancy_color_coded)
            ], axis=-1).astype(np.uint8)

            cv2.circle(occupancy_color_coded, (args.resolution//2, args.resolution//2), 3,(255, 0, 0), -1)
            images.append(occupancy_color_coded)
            

            # edge detection
            laplacian = cv2.Laplacian(occupancy_probability,cv2.CV_32F)
            laplacian = cv2.cvtColor(laplacian, cv2.COLOR_GRAY2RGB)
            images.append(laplacian.astype(np.uint8))

        if args.round_grid:
            rings, degrees = init_round_grid(
                cartesian_resolution=args.resolution,
                angle_resolution=args.angle_resolution,
                ring_width=args.ring_width, 
                growth_rate=args.growth_rate)
            
            radar_map = discretize_radar(sensor_data=sensor_data, rings=rings, angles=degrees)
            lidar_map = discretize_lidar(sensor_data=sensor_data, rings=rings, angles=degrees)

            # dempster shafer theory
            occupancy_probability = dempster_shafer_fusion(
                lidar_grid=lidar_map,
                radar_grid=radar_map,
                weights=dempster_shafer_parameter
            )


            occupancy_probability *=255
            occupancy_probability.astype(np.uint8)
            occupancy_probability = polar_coordinates_to_cartesian_grid(
                polar_grid=occupancy_probability, 
                cartesian_resolution=args.resolution,
                rings=rings,
                )
            
            lidar_map = polar_coordinates_to_cartesian_grid(
                polar_grid=lidar_map, 
                cartesian_resolution=args.resolution,
                rings=rings,
                )
            
            lidar_map = cv2.resize(lidar_map, (args.resolution, args.resolution), interpolation=cv2.INTER_NEAREST_EXACT)
            lidar_map = lidar_map.T
            lidar_map = cv2.cvtColor((lidar_map*255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
            cv2.circle(lidar_map, (args.resolution//2, args.resolution//2), 3, (255,0,0), -1)
            # images.append(lidar_map)

            occupancy_probability = occupancy_probability.T.copy()
            occupancy_color_coded = np.expand_dims(occupancy_probability, axis=-1)
            occupancy_color_coded = np.concatenate([
                np.zeros_like(occupancy_color_coded),
                occupancy_color_coded,
                np.zeros_like(occupancy_color_coded)
            ], axis=-1).astype(np.uint8)

            # edge detection
            laplacian = cv2.Laplacian(occupancy_probability.astype(np.float32),cv2.CV_32F)
            laplacian = cv2.cvtColor(laplacian, cv2.COLOR_GRAY2RGB).astype(np.uint8)

            cv2.circle(occupancy_color_coded, (args.resolution//2, args.resolution//2), 3, (255,0,0), -1)
            images.append(occupancy_color_coded)
            
            cv2.circle(laplacian, (args.resolution//2, args.resolution//2), 3, (255,0,0), -1)
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
        show_image(image=frame, time=args.time)

        # Get next sample
        sample = nusc.get('sample', sample["next"])

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
