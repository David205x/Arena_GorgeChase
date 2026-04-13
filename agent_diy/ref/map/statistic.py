import json
import os
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt


MAP_SIZE = 128
MAP_DIR = r'D:\_Project\Python\Aiarena\2026\Arena_GorgeChase\agent_diy\ref\map'
# MAP_DIR = '/data/projects/gorge_chase/agent_diy/ref/map'


def json_to_ndarray(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    cells = data.get('cells', [])
    matrix = np.zeros((MAP_SIZE, MAP_SIZE))
    for cell in cells:
        if cell['type_id'] == 1:
            matrix[cell['z']][cell['x']] = 1
    return matrix


def draw_map(_id: str, matrix: np.ndarray):
    """
    将matrix使用matplotlib绘制保存为名称为f'map_{_id}.jpg'的二值图像,
    matrix为128*128的大小，保证输出图像为正方形，横纵坐标步长一致
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(matrix, cmap='gray', vmin=0, vmax=1, origin='lower', interpolation='nearest')
    ax.set_aspect('equal')
    ax.set_xticks(np.arange(0, MAP_SIZE + 1, 16))
    ax.set_yticks(np.arange(0, MAP_SIZE + 1, 16))
    ax.set_xlim(-0.5, MAP_SIZE - 0.5)
    ax.set_ylim(-0.5, MAP_SIZE - 0.5)
    ax.set_title(f'Map {_id}')
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(f'map_{_id}.jpg', dpi=200, bbox_inches='tight')
    plt.close(fig)


if __name__ == "__main__":
    maps = {}
    for file in os.listdir(MAP_DIR):
        if file.startswith('gorge_chase_map_'):
            map_id = file.split('_')[3].split('.')[0]
            _map = json_to_ndarray(os.path.join(MAP_DIR, file))
            draw_map(map_id, _map)
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