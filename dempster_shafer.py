import numpy as np
from typing import Dict


def dempster_shafer_fusion(lidar_grid: np.ndarray, radar_grid: np.ndarray, weights: Dict[str, float]) -> np.ndarray:
    lidar_mass_values = mass_function(
        cells=lidar_grid,
        weight_occupied=weights["lidar_occupied"],
        weight_unoccupied=weights["lidar_unoccupied"],
        weight_unsure=weights["lidar_unsure"],
    )

    radar_mass_values = mass_function(
        cells=radar_grid,
        weight_occupied=weights["radar_occupied"],
        weight_unoccupied=weights["radar_unoccupied"],
        weight_unsure=weights["radar_unsure"],
    )

    cell_occupancy = combination_rule(
        mass_values_1=lidar_mass_values,
        mass_values_2=radar_mass_values
    )

    return cell_occupancy


def mass_function(cells: np.ndarray, weight_occupied: float, weight_unoccupied: float, weight_unsure: float) -> Dict[str,np.ndarray]:
    mass_values = {}
    mass_values["occupied"] = weight_occupied * cells
    mass_values["unoccupied"] = weight_unoccupied * cells
    mass_values["uncertain"] = weight_unsure * cells
    # return np.concatenate([occupied, unoccupied, unsure], axis=-1)
    return mass_values



def combination_rule(mass_values_1: Dict[str,np.ndarray], mass_values_2: Dict[str,np.ndarray]) -> np.ndarray:
    numerator = mass_values_1["occupied"] * mass_values_2["occupied"]
    numerator += mass_values_1["uncertain"] * mass_values_2["occupied"]
    numerator += mass_values_1["occupied"] * mass_values_2["uncertain"]

    denominator = 1 - mass_values_1["unoccupied"] * mass_values_2["occupied"]
    denominator -= mass_values_1["occupied"] * mass_values_2["unoccupied"]
    
    return np.divide(numerator, denominator + 1e-8)