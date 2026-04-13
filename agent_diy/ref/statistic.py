import json
import os
import numpy as np


MAP_SIZE = 128
MAP_DIR = '/data/projects/gorge_chase/agent_diy/ref'


def json_to_ndarray(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    cells = data.get('cells', [])
    matrix = np.zeros((MAP_SIZE, MAP_SIZE))
    for cell in cells:
        if cell['type_id'] == 1:
            matrix[cell['z']][cell['x']] = 1
    return matrix


if __name__ == "__main__":
    maps = {}
    for file in os.listdir(MAP_DIR):
        if file.startswith('gorge_chase_map_'):
            map_id = file.split('_')[3].split('.')[0]
            _map = json_to_ndarray(os.path.join(MAP_DIR, file))
            count_0 = np.count_nonzero(_map == 0) 
            count_1 = np.count_nonzero(_map == 1)
            maps[map_id] = {
                'count_0': count_0,
                'count_1': count_1,
                'percent_0': count_0 / (count_0 + count_1),
                'percent_1': count_1 / (count_0 + count_1),
            }
    with open('maps_statistic.json', 'w') as f:
        json.dump(maps, f, indent=4)