import numpy as np


def compute_length_and_degree(pcd):
        vector_length = np.linalg.norm(pcd, ord=2, axis=-1)
        pcd_unit_vector = np.divide(pcd, np.repeat(vector_length.reshape((-1, 1)), 3, axis=-1))
        sin = np.rad2deg(np.arcsin(pcd_unit_vector[:, 1]))
        cos = np.rad2deg(np.arccos(pcd_unit_vector[:, 0]))
        quadrant = np.select([((sin > 0) & (cos < 0)), ((sin < 0) & (cos < 0)), ((sin < 0) & (cos > 0))], [90, 180, 270], default=0)
        # print(np.concatenate([pcd[:, :2], np.column_stack([sin, cos, quadrant])], axis=-1))
        
        degree = quadrant + np.abs(cos)
        return vector_length, degree