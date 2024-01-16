#!/bin/bash
python stream_occupancy_grid.py \
--scene_number 2 \
--round_grid \
--angle_resolution 2 \
--ring_width 1.0 \
--growth_rate 1.0 \
--lidar_occupied 1.0 \
--lidar_unoccupied 0.0 \
--lidar_unsure 0.0 \
--radar_occupied 0.5 \
--radar_unoccupied 0.0 \
--radar_unsure .5 \
--time 500