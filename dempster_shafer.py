import numpy as np



def dempster_shafer_theory(lidar_grid: np.ndarray, radar_grid: np.ndarray, m1_theta: float, m2_theta: float) -> np.ndarray:
    # m1(B): lidar occupied; m2(B): radar occupied

    # numerator = m1(B) * m2(B) + m1(theta) * m2(B) + m1(B) * m2(theta)
    numerator = lidar_grid * radar_grid + m1_theta * radar_grid + m2_theta * lidar_grid
    # denominator = 1 - m1(B') * m2(B) -  m1(B) * m2(B')
    denominator = 1 - np.subtract(1, lidar_grid) * radar_grid - lidar_grid * np.subtract(1, radar_grid) 
    
    return np.divide(numerator, denominator)