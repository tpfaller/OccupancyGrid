# Square Grid

```
python stream_occupancy_grid.py \
--scene_number 2 \
--max_height 2.5 \
--square_grid \
--viz_raw_pointcloud \
--lidar_occupied 0.7 \
--lidar_unoccupied 0.1 \
--lidar_unsure 0.2 \
--radar_occupied 0.7 \
--radar_unoccupied 0.1 \
--radar_unsure 0.2 
```

# Square Grid + Exact Sensor

```
python stream_occupancy_grid.py \
--scene_number 2 \
--max_height 2.5 \
--square_grid \
--viz_raw_pointcloud \
--lidar_occupied 1.0 \
--lidar_unoccupied 0.0 \
--lidar_unsure 0.0 \
--radar_occupied 1.0 \
--radar_unoccupied 0.0 \
--radar_unsure 0.0 \
--resolution 150
```

# Square Grid + Most points

```
python stream_occupancy_grid.py \
--scene_number 2 \
--max_height 3.5 \
--square_grid \
--viz_raw_pointcloud \
--lidar_occupied 0.5 \
--lidar_unoccupied 0.0 \
--lidar_unsure 0.5 \
--radar_occupied 0.5 \
--radar_unoccupied 0.0 \
--radar_unsure 0.5 \
--resolution 150
```


# Round Grid

```
python stream_occupancy_grid.py \
--scene_number 2 \
--max_height 2.5 \
--round_grid \
--viz_raw_pointcloud \
--lidar_occupied 0.7 \
--lidar_unoccupied 0.1 \
--lidar_unsure 0.2 \
--radar_occupied 0.7 \
--radar_unoccupied 0.1 \
--radar_unsure 0.2 
```

# Round Grid + Exact Sensors 

```
python stream_occupancy_grid.py \
--scene_number 2 \
--max_height 2.5 \
--round_grid \
--viz_raw_pointcloud \
--lidar_occupied 1.0 \
--lidar_unoccupied 0.0 \
--lidar_unsure 0.0 \
--radar_occupied 1.0 \
--radar_unoccupied 0.0 \
--radar_unsure 0.0
```

# Round Grid + Most Points

```
python stream_occupancy_grid.py \
--scene_number 2 \
--max_height 3.5 \
--min_height 0.1 \
--min_x 0.5 \
--min_y 0.5 \
--round_grid \
--viz_raw_pointcloud \
--lidar_occupied 0.5 \
--lidar_unoccupied 0.0 \
--lidar_unsure 0.5 \
--radar_occupied 0.5 \
--radar_unoccupied 0.0 \
--radar_unsure 0.5 \
--resolution 300 \
--ring_width .25 \
--growth_rate 1.0 \
--angle_resolution 1
```
