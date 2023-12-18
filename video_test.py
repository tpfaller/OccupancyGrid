from nuscenes.nuscenes import NuScenes
import numpy as np
import cv2

from pprint import pprint

from occupancy_grid import OccupancyGrid
from pointcloud_tools import *

def video_radar_pointcloud():
    # Get scene and first sample
    data_path = "data/v1.0-mini"
    nusc = NuScenes(version='v1.0-mini', dataroot=data_path, verbose=False)
    my_scene = nusc.scene[0]
    my_sample = nusc.get('sample', my_scene['first_sample_token'])

    # Init the occupancy grid
    occupancy_grid = OccupancyGrid((640, 640, 3))
    sensor_list = ["RADAR_FRONT", "RADAR_FRONT_LEFT", "RADAR_BACK_LEFT", "RADAR_BACK_RIGHT", "RADAR_FRONT_RIGHT", "LIDAR_TOP"]

    while len(my_sample["next"]) > 0:
        # Let something happen here
        frames = []
        frames_ego = []

        for sensor in sensor_list:
            pcd, pcd_ego = get_calibrated_pointcloud(nusc, my_sample, sensor_name=sensor)
            pcd *=3
            pcd_ego *= 3
            bev = occupancy_grid.draw_pointcloud_in_birds_eye_view(pcd)
            frames.append(bev)

            bev_ego = occupancy_grid.draw_pointcloud_in_birds_eye_view(pcd_ego)
            frames_ego.append(bev_ego)

        uncalibrated_radar_1 = np.concatenate(frames[:len(frames)//2], axis=1)
        uncalibrated_radar_2 = np.concatenate(frames[len(frames)//2:], axis=1)
        uncalibrated_radar = np.concatenate([uncalibrated_radar_1, uncalibrated_radar_2], axis=0)
        ego_pose_radar_1 = np.concatenate(frames_ego[:len(frames_ego)//2], axis=1)
        ego_pose_radar_2 = np.concatenate(frames_ego[len(frames_ego)//2:], axis=1)
        ego_pose_radar = np.concatenate([ego_pose_radar_1, ego_pose_radar_2], axis=0)
        cv2.namedWindow("Uncalibrated Radar", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Uncalibrated Radar", 3 *640, 640) 
        cv2.imshow("Uncalibrated Radar", uncalibrated_radar)

        cv2.namedWindow("Ego Pose Radar", cv2.WINDOW_NORMAL) 
        cv2.resizeWindow("Ego Pose Radar", 3 * 640, 640) 
        cv2.imshow("Ego Pose Radar", ego_pose_radar)

        cv2.waitKey(0)

        # Get the next sample
        my_sample = nusc.get('sample', my_sample["next"])
    cv2.destroyAllWindows()

def video_lidar_occupancy_grid():
    # Get scene and first sample
    data_path = "data/v1.0-mini"
    nusc = NuScenes(version='v1.0-mini', dataroot=data_path, verbose=False)
    my_scene = nusc.scene[0]
    my_sample = nusc.get('sample', my_scene['first_sample_token'])

    # Init grid structure
    angle_resolution = 3
    ring_width, growth_rate = .25, 1.025
    rings = get_rings(ring_width=ring_width, growth_rate=growth_rate)
    angles = np.arange(0, 360, angle_resolution)

    while len(my_sample["next"]) > 0:
        _, lidar_pcd_ego = get_calibrated_pointcloud(nusc, my_sample, sensor_name="LIDAR_TOP")
        lidar_pcd_ego = filter_pointcloud(lidar_pcd_ego)

        points_per_cell = count_points_per_cell(pcd=lidar_pcd_ego, angles=angles, circles=rings)
        occupied = np.select([points_per_cell > 0], [1.], default=0.)

        cv2.namedWindow("Occupancy Lidar", cv2.WINDOW_NORMAL) 
        cv2.resizeWindow("Occupancy Lidar", 1280, 1280) 
        cv2.imshow("Occupancy Lidar", occupied)

        cam_front = nusc.get('sample_data', my_sample['data']['CAM_FRONT'])
        file_path = os.path.join("data/v1.0-mini/", cam_front["filename"])
        image_front = cv2.imread(file_path)
        cv2.namedWindow("Image Front", cv2.WINDOW_NORMAL) 
        cv2.resizeWindow("Image Front", 1280, 1280) 
        cv2.imshow("Image Front", image_front)

        cv2.waitKey(0)

        # Get the next sample
        my_sample = nusc.get('sample', my_sample["next"])
    cv2.destroyAllWindows()




def main():
    # test_video()
    # nu_camera_video()
    # video_radar_pointcloud()
    video_lidar_occupancy_grid()


if __name__ == '__main__':
    main()