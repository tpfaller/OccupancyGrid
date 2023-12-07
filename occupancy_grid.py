import math

import numpy as np
import cv2

from utils import read_pcd_file

class OccupancyGrid:
    def __init__(self, shape: tuple) -> None:
        self.angle_resolution = 3

        self.grid = np.zeros(shape=shape)
        self.init_birds_eye_view()

    def init_birds_eye_view(self):
        shape = (640, 640, 3)
        self.center = (shape[0]//2, shape[1]//2)
        img = np.zeros(shape)
        for exponent in range(4, int(math.log2(shape[0])) + 1):
            radius = 2 ** exponent
            cv2.circle(img,self.center, radius, (255,255,255), thickness=1)

        for degree in range(0, 180, 10):
            radian = math.radians(degree)
            vector_length = 320
            x = int(math.cos(radian) * vector_length)
            y = int(math.sin(radian) * vector_length)
            top_left =  self.center[0] - x, self.center[1] - y
            bottom_right =  self.center[0] + x, self.center[1] + y
            cv2.line(img, top_left, bottom_right, color=(255, 255, 255), thickness=1)
        self.img = img


    def test_point_cloud(self):
        path = "data/v1.0-trainval01_blobs_radar/samples/RADAR_BACK_LEFT/n008-2018-08-01-15-16-36-0400__RADAR_BACK_LEFT__1533151061567861.pcd"
        pcd = read_pcd_file(path)

    def viz_birds_eye_view(self):
        cv2.imshow("Birdseye_View", self.img)
        cv2.waitKey(0) 
        cv2.destroyAllWindows()

    def viz_pointcloud_in_birds_eye_view(self, path):
        pcd = read_pcd_file(path)
        pcd = pcd.astype(np.int32)[:, :2]
        pcd = np.add(pcd, self.center)
        bev = self.img.copy()
        for pt in pcd:
            cv2.circle(bev, pt, radius=2, color=(255, 0, 0), thickness=-1)
        cv2.imshow("Birdseye_View", bev)
        cv2.waitKey(0) 
        cv2.destroyAllWindows()


def main():
    grid = OccupancyGrid(shape=(12, 12))
    # grid.test_point_cloud()
    # grid.viz_birds_eye_view()
    path = "data/v1.0-trainval01_blobs_radar/samples/RADAR_BACK_LEFT/n008-2018-08-01-15-16-36-0400__RADAR_BACK_LEFT__1533151061567861.pcd"
    grid.viz_pointcloud_in_birds_eye_view(path=path)

    path = "data/v1.0-trainval01_blobs_radar/samples/RADAR_BACK_RIGHT/n008-2018-08-01-15-16-36-0400__RADAR_BACK_RIGHT__1533151061554664.pcd"
    grid.viz_pointcloud_in_birds_eye_view(path=path)

    path = "data/v1.0-trainval01_blobs_radar/samples/RADAR_FRONT/n008-2018-08-01-15-16-36-0400__RADAR_FRONT__1533151061553525.pcd"
    grid.viz_pointcloud_in_birds_eye_view(path=path)

    
if __name__ == '__main__':
    main()
