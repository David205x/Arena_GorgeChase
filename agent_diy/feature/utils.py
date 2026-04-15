import numpy as np

from .constant import *


def chebyshev_distance(x1: int, z1: int, x2: int, z2: int) -> int:
    return max(abs(x1 - x2), abs(z1 - z2))


def is_pos_neighbor(x1: int, z1: int, x2: int, z2: int) -> bool:
    if chebyshev_distance(x1, z1, x2, z2) <= 1:
        return True
    return False


def clamp_map_coord(x: int, z: int) -> tuple[int, int]:
    return min(max(x, 0), MAP_SIZE - 1), min(max(z, 0), MAP_SIZE - 1)


def predict_flash_pos(map_view: np.ndarray, x: int, z: int) -> list[tuple[int, int]]:
    """
    预测 8 方向的闪现落点。

    :param map_view: 21*21 的局部地图，索引为 map_view[z, x]
    :param x: 当前英雄在局部图中的列索引
    :param z: 当前英雄在局部图中的行索引
    :return: 按 [右, 右上, 上, 左上, 左, 左下, 下, 右下] 顺序返回落点坐标
    """
    if map_view.ndim != 2 or map_view.shape != (VIEW_SIZE, VIEW_SIZE):
        raise ValueError("map_view 须为 21x21 的二维矩阵")

    def walkable(nx: int, nz: int) -> bool:
        if nx < 0 or nx >= VIEW_SIZE or nz < 0 or nz >= VIEW_SIZE:
            return False
        return bool(map_view[nz, nx] == 1)

    out: list[tuple[int, int]] = []
    for dx, dz in FLASH_DIR_VEC:
        is_diagonal = dx != 0 and dz != 0
        max_step = FLASH_DISTANCE_DIAGONAL if is_diagonal else FLASH_DISTANCE
        landed: tuple[int, int] | None = None
        for step in range(max_step, 0, -1):
            nx = x + dx * step
            nz = z + dz * step
            if walkable(nx, nz):
                landed = (nx, nz)
                break
        if landed is None:
            landed = (x, z)
        out.append(landed)
    return out


def flash_pos_relative(flash_pos: list[tuple[int, int]], x: int, z: int) -> list[tuple[int, int]]:
    """
    将局部视野中的闪现落点坐标转为相对位移 (dx, dz)。
    """
    return [(land_x - x, land_z - z) for land_x, land_z in flash_pos]


def flash_validation(flash_pos_relative: list[tuple[int, int]]) -> list[bool]:
    """
    根据相对位移判断各方向闪现是否产生有效位移。
    """
    return [bool(dz != 0 or dx != 0) for dx, dz in flash_pos_relative]


def distance_l2(x1: int, z1: int, x2: int, z2: int) -> float:
    return float(np.hypot(x1 - x2, z1 - z2))


def build_local_window(global_map: np.ndarray, x: int, z: int, pad_value: int = -1) -> np.ndarray:
    assert global_map.ndim == 2, "global_map 须为二维矩阵"

    h, w = global_map.shape
    assert h == w == MAP_SIZE, f"global_map 须为 {MAP_SIZE}x{MAP_SIZE} 的矩阵"

    out = np.full((VIEW_SIZE, VIEW_SIZE), pad_value, dtype=global_map.dtype)

    x_min = x - VIEW_CENTER
    x_max = x + VIEW_CENTER + 1
    z_min = z - VIEW_CENTER
    z_max = z + VIEW_CENTER + 1

    global_x_start = max(0, x_min)
    global_x_end = min(MAP_SIZE, x_max)
    global_z_start = max(0, z_min)
    global_z_end = min(MAP_SIZE, z_max)

    view_x_start = max(0, -x_min)
    view_x_end = VIEW_SIZE - max(0, x_max - MAP_SIZE)
    view_z_start = max(0, -z_min)
    view_z_end = VIEW_SIZE - max(0, z_max - MAP_SIZE)

    out[view_z_start:view_z_end, view_x_start:view_x_end] = global_map[
        global_z_start:global_z_end,
        global_x_start:global_x_end,
    ]
    return out
