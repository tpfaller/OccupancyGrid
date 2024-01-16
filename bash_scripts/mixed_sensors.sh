#!/bin/bash
python stream_occupancy_grid.py \
--scene_number 2 \
--viz_raw_pointcloud \
--round_grid \
--angle_resolution 1 \
--ring_width 1.0 \
--growth_rate 1.025 \
--lidar_occupied 0.7 \
--lidar_unoccupied 0.0 \
--lidar_unsure 0.3 \
--radar_occupied 0.7 \
--radar_unoccupied 0.0 \
--radar_unsure 0.3 \
--time 0