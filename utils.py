import numpy as np 
import open3d as o3d


def read_pcd_file(file_path):
    """
    Reads in a PCD file and returns a numpy array of the point cloud data.
    """
    pcd = o3d.io.read_point_cloud(file_path)
    out_arr = np.asarray(pcd.points)  
    return out_arr