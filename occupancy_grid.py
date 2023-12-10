import math

import numpy as np
import cv2

from pointcloud_tools import read_pcd_file, read_bin_file, compute_length_and_degree

class OccupancyGrid:
    def __init__(self, shape: tuple=(640, 640, 3)) -> None:
        self.angle_resolution = 15
        self.init_birds_eye_view(shape)

    def init_birds_eye_view(self, shape: tuple):
        self.center = (shape[0]//2, shape[1]//2)
        img = np.zeros(shape, dtype=np.uint8)
        
        center = 12
        cell_wide = 16
        self.circles = []

        radius = center + cell_wide
        while radius < min(shape[:2]) // 2:
            self.circles.append(radius)
            cv2.circle(img,self.center, radius, (255,255,255), thickness=2)
            cell_wide = int(cell_wide * 1.25)
            radius += cell_wide

        for degree in range(0, 180, self.angle_resolution):
            radian = math.radians(degree)
            vector_length = self.circles[-1]
            x = int(math.cos(radian) * vector_length)
            y = int(math.sin(radian) * vector_length)
            top_left =  self.center[0] - x, self.center[1] - y
            bottom_right =  self.center[0] + x, self.center[1] + y
            cv2.line(img, top_left, bottom_right, color=(255, 255, 255), thickness=1)

        cv2.circle(img,self.center, center, (0,255, 0), thickness=-1)

        self.img = img

    def viz_birds_eye_view(self):
        cv2.imshow("Birdseye_View", self.img)
        cv2.waitKey(0) 
        cv2.destroyAllWindows()

    def viz_pointcloud_in_birds_eye_view(self, path):
        if path[-3:] == "bin":
            pcd = read_bin_file(path)
        else:
            pcd = read_pcd_file(path)
        
        bev = self.draw_pointcloud_in_birds_eye_view(pcd)
        self.count_points_per_cell(pcd)

        cv2.imshow("Birdseye_View", bev)
        cv2.waitKey(0) 
        cv2.destroyAllWindows()

    def draw_pointcloud_in_birds_eye_view(self, pcd: np.ndarray) -> np.ndarray:
        # Discard height and reflection
        pcd = pcd.astype(np.int32)[:, :2]

        # Shift pointcloud into the mid of the image
        pcd = np.add(pcd, self.center)
        bev = self.img.copy()
        for pt in pcd:
            cv2.circle(bev, pt, radius=2, color=(255, 0, 0), thickness=-1)
        return bev


    def count_points_per_cell(self, pcd: np.ndarray) -> np.ndarray:
        # Bin for angles
        angles = np.arange(start=0, stop=361, step=self.angle_resolution, dtype=np.int32)

        vector_lengths, pcd_angles = compute_length_and_degree(pcd)
    
        angle_bins = np.digitize(pcd_angles, angles)
        radius_bins = np.digitize(vector_lengths, self.circles)

        print(angle_bins)
        print(radius_bins)


    def draw_occupancy_in_birds_eye_view(self, pcd: np.ndarray) -> np.ndarray:
        pcd = pcd.astype(np.int32)[:, :2]
        pcd = np.add(pcd, self.center)
        bev = self.img.copy()
        for pt in pcd:
            cv2.circle(bev, pt, radius=2, color=(255, 0, 0), thickness=-1)
        return bev

def main():
    grid = OccupancyGrid(shape=(640, 640, 3))
    # grid.test_point_cloud()
    # grid.viz_birds_eye_view()
    # path = "data/v1.0-mini/samples/RADAR_BACK_LEFT/n008-2018-08-01-15-16-36-0400__RADAR_BACK_LEFT__1533151603522238.pcd"
    # grid.viz_pointcloud_in_birds_eye_view(path=path)

    path = "data/v1.0-mini/samples/RADAR_FRONT/n008-2018-08-01-15-16-36-0400__RADAR_FRONT__1533151603555991.pcd"
    grid.viz_pointcloud_in_birds_eye_view(path=path)

    # path = "data/v1.0-mini/samples/LIDAR_TOP/n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151603547590.pcd.bin"
    # grid.viz_pointcloud_in_birds_eye_view(path=path)

    
if __name__ == '__main__':
    main()
