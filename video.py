import os
from pprint import pprint

import numpy as np
import cv2

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import transform_matrix
from pyquaternion import Quaternion

from pointcloud_tools import read_bin_file, read_pcd_file
from occupancy_grid import OccupancyGrid


def test_video():
    black = np.zeros((640, 640, 3), dtype=np.uint8)
    contrast = np.concatenate(
        [
            np.zeros((320, 640, 3), dtype=np.uint8), 
            np.ones((320, 640, 3), dtype=np.uint8) * 10
        ],
        axis=0
        )
    
    for i in range(20): 
        black += contrast
        cv2.imshow("test", black)
        cv2.waitKey(100)
    cv2.destroyAllWindows()

def nu_camera_video():
    data_path = "data/v1.0-mini"

    nusc = NuScenes(version='v1.0-mini', dataroot='data/v1.0-mini', verbose=True)
    my_scene = nusc.scene[0]


    first_sample_token = my_scene['first_sample_token']
    my_sample = nusc.get('sample', first_sample_token)


    my_camera = nusc.get('sample_data', my_sample['data']['CAM_FRONT'])
    
    while len(my_camera["next"]) > 0:
        if my_camera['is_key_frame']:
            image_path = os.path.join(data_path, my_camera["filename"])
            image = cv2.imread(image_path)
            cv2.imshow("Camera Front", image)
            cv2.waitKey(500)

        my_camera = nusc.get('sample_data', my_camera["next"])

    cv2.destroyAllWindows()

def stream_scene():
    sensors = ["CAM_FRONT", "RADAR_FRONT"] # "LIDAR_TOP"] # , "RADAR_FRONT"]

    # Get scene and first sample
    data_path = "data/v1.0-mini"
    nusc = NuScenes(version='v1.0-mini', dataroot='data/v1.0-mini', verbose=True)
    my_scene = nusc.scene[0]
    my_sample = nusc.get('sample', my_scene['first_sample_token'])

    # Init the occupancy grid
    occupancy_grid = OccupancyGrid((640, 640, 3))

    calibration_matrices = {}
    for sensor in sensors:
        sensor_token = my_sample["data"][sensor]
        sample_data = nusc.get('sample_data', sensor_token)
        calibration_data = nusc.get("calibrated_sensor", sample_data['calibrated_sensor_token'])
        pprint(calibration_data)

        P = transform_matrix(
            translation=np.array(calibration_data["translation"]),
            rotation=Quaternion(calibration_data["rotation"])
        )
        calibration_matrices[sensor_token] = P

    # Stream the scene samples
    while len(my_sample["next"]) > 0:
        frames = list()

        for sensor in sensors:
            sensor_token = my_sample["data"][sensor]

            filename = nusc.get('sample_data', sensor_token)["filename"]
            file_path = os.path.join(data_path, filename)

            if file_path[-3:] == "bin":
                pcd = read_bin_file(file_path)
                pcd = pcd[np.where(pcd[:, 2] < 0)][:,:2]
                pcd *= 3
                frames.append(occupancy_grid.draw_pointcloud_in_birds_eye_view(pcd=pcd))
            elif file_path[-3:] == "pcd":
                pcd = read_pcd_file(file_path)
                pcd *= 3
                frames.append(occupancy_grid.draw_pointcloud_in_birds_eye_view(pcd=pcd))
                print(f"Minimum : {np.min(pcd):4.2f}\tMaximum : {np.max(pcd):.2f}")
            else:
                image = cv2.imread(file_path)
                
                image = cv2.resize(image, (1280, 640))
                frames.append(image)
            
        # Put all the frames together and display at once
        data_stream = np.concatenate(frames, axis=1)
        cv2.namedWindow("All sensors", cv2.WINDOW_NORMAL) 
        cv2.resizeWindow("All sensors", 1280, 640) 
        cv2.imshow("All sensors", data_stream)
        cv2.waitKey(0)
            
        # Get the next sample
        my_sample = nusc.get('sample', my_sample["next"])
    cv2.destroyAllWindows()


def main():
    # test_video()
    # nu_camera_video()
    stream_scene()


if __name__ == '__main__':
    main()
