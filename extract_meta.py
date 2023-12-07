import os
import json
from pprint import pprint


def get_all_radar_scenes(path: str):
    sensors = [os.path.join(path, x) for x in os.listdir(path)]

    scenes, timestamps = [], []

    for sensor in sensors:
        all_samples = os.listdir(sensor)
        
        for sample in all_samples:
            scene, _, _= sample.split('__')
            scenes.append(scene)
            # timestamps.append(timestamp)

    return set(scenes)
    
def open_samples_meta(path: str):
    with open(path, "r") as f:
        samples_file = json.load(f)
    print(len(samples_file))
    pprint(samples_file[:4])

def collect_log_files(path: str):
    with open(path, "r") as f:
        samples_file = json.load(f)
    print(len(samples_file))
    pprint(samples_file[:4])


def main():
    # radar_samples = "data/v1.0-trainval01_blobs_radar/samples"
    # get_all_radar(radar_samples)
    open_samples_meta("data/v1.0-trainval_meta/v1.0-trainval/log.json")



if __name__ == '__main__':
    main()
