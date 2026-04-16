# Map size / 地图尺寸（128×128）
MAP_SIZE = 128
# Local view size / 局部视野尺寸（21×21）
VIEW_SIZE = 21
# Local view center index / 局部视野中心索引
VIEW_CENTER = VIEW_SIZE // 2
# Flash max distance / 闪现直线最大距离
FLASH_DISTANCE = 10
# Flash diagonal max distance / 闪现斜向最大距离
FLASH_DISTANCE_DIAGONAL = 8
# Position history length / 历史位置缓存长度
POS_HISTORY_LEN = 10


FLASH_DIR_VEC: list[tuple[int, int]] = [
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
]
"""[右, 右上, 上, 左上, 左, 左下, 下, 右下]"""

MOVE_DIR_VEC: list[tuple[int, int]] = [
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
]
"""与移动 8 方向保持一致"""

ALPHA_MAP = {
    1: 0.5,
    2: 0.6,
    3: 0.90,
}
