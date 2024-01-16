#!/bin/bash
python stream_occupancy_grid.py \
--scene_number 2 \
--viz_raw_pointcloud \
--square_grid \
--lidar_occupied 1.0 \
--lidar_unoccupied 0.0 \
--lidar_unsure 0.0 \
--radar_occupied 1.0 \
--radar_unoccupied 0.0 \
--radar_unsure 0.0 \
--time 0