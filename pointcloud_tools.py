import numpy as np
import open3d as o3d


def read_pcd_file(file_path):
    """
    Reads in a PCD file and returns a numpy array of the point cloud data.
    """
    pcd = o3d.io.read_point_cloud(file_path)
    out_arr = np.asarray(pcd.points)  
    return out_arr


def compute_length_and_degree(pcd):
    vector_length = np.linalg.norm(pcd, ord=2, axis=-1)
    pcd_unit_vector = np.divide(pcd, np.repeat(vector_length.reshape((-1, 1)), 3, axis=-1))
    sin = np.rad2deg(np.arcsin(pcd_unit_vector[:, 1]))
    cos = np.rad2deg(np.arccos(pcd_unit_vector[:, 0]))
    quadrant = np.select([((sin > 0) & (cos < 0)), ((sin < 0) & (cos < 0)), ((sin < 0) & (cos > 0))], [90, 180, 270], default=0)
    # print(np.concatenate([pcd[:, :2], np.column_stack([sin, cos, quadrant])], axis=-1))

    degree = quadrant + np.abs(cos)
    return vector_length, degree

def read_bin_file(file_path):
    scan = np.fromfile(file_path, dtype=np.float32)
    points = scan.reshape((-1, 5))[:, :4]
    return points