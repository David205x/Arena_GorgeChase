import numpy as np
from collections import deque


# Map size / 地图尺寸（128×128）
MAP_SIZE = 128.0
# Max monster speed / 最大怪物速度
MAX_MONSTER_SPEED = 5.0
# Max distance bucket / 距离桶最大值
MAX_DIST_BUCKET = 5.0
# Max flash cooldown / 最大闪现冷却步数
MAX_FLASH_CD = 2000.0
# Max buff duration / buff最大持续时间
MAX_BUFF_DURATION = 50.0


_FLASH_DIR_VECS: list[tuple[int, int]] = [
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
]

def predict_flash_pos(map_view: np.ndarray, x: int, z: int) -> list[tuple[int, int]]:
    """
    预测 8 方向的闪现落点（与 Lua execute_flash：从最远步递减到 1，取第一个可走格；否则原地）。

    :param map_view: 21*21 的矩阵，1 为可走，0 为不可走；索引为 map_view[z, x]。
    :param x: 玩家在局部图中的列索引（水平）。
    :param z: 玩家在局部图中的行索引（竖直）。
    :return: 按 [右、右上、上、左上、左、左下、下、右下] 顺序，各方向落点 (x, z)。

    直线最大步数 10、斜向最大步数 8
    """
    if map_view.ndim != 2 or map_view.shape[0] != map_view.shape[1]:
        raise ValueError("map_view 须为方阵（通常为 21x21）")
    h, w = map_view.shape

    flash_distance = 10
    flash_distance_diagonal = 8

    def walkable(nx: int, nz: int) -> bool:
        if nx < 0 or nx >= w or nz < 0 or nz >= h:
            return False
        return bool(map_view[nz, nx] == 1)

    out: list[tuple[int, int]] = []
    for dx, dz in _FLASH_DIR_VECS:
        is_diagonal = dx != 0 and dz != 0
        max_step = flash_distance_diagonal if is_diagonal else flash_distance
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

    :param flash_pos: `predict_flash_pos` 返回的 8 个方向落点，元素为局部坐标 (x, z)
    :param x: 当前玩家在局部图中的列索引
    :param z: 当前玩家在局部图中的行索引
    :return: 按 [右、右上、上、左上、左、左下、下、右下] 顺序返回相对位移 (dx, dz)
    """
    return [(land_x - x, land_z - z) for land_x, land_z in flash_pos]



def flash_validation(flash_pos_relative: list[tuple[int, int]]) -> list[bool]:
    """
    根据相对位移判断各方向闪现是否产生有效位移。

    只要相对位移不是 (0, 0)，就说明该方向不会“原地闪现”。
    """
    return [bool(dz != 0 or dx != 0) for dx, dz in flash_pos_relative]



def chebyshev_distance(x1: int, z1: int, x2: int, z2: int) -> int:
    return max(abs(x1 - x2), abs(z1 - z2))


def is_pos_neighbor(x1: int, z1: int, x2: int, z2: int) -> bool:
    if chebyshev_distance(x1, z1, x2, z2) <= 1:
        return True
    return False


class Character:
    def __init__(self, obs: dict):
        self.id: int = obs.get('hero_id', 0) or obs.get('monster_id', 0) or obs.get('config_id', 0)
        self.x: int = obs['pos']['x']
        """[0, 127]"""
        self.z: int = obs['pos']['z']
        """[0, 127]"""
        self.hero_l2_distance: int = obs.get('hero_l2_distance', 0)
        """
        与英雄的欧氏距离桶编号 (0-5), 均匀划分.
        128×128 地图均匀划分：
        0=[0,30), 1=[30,60), 
        2=[60,90), 3=[90,120), 
        4=[120,150), 5=[150,180]
        """
        self.hero_relative_direction: int = obs.get('hero_relative_direction', 0)
        """
        相对于英雄的方位 (0-8)
        0=重叠/无效，
        1=东，2=东北，
        3=北，4=西北，
        5=西，6=西南，
        7=南，8=东南
        """


class Hero(Character):
    def __init__(self, obs: dict):
        super(Hero, self).__init__(obs)
        self.buff_remaining_time: int = obs['buff_remaining_time']
        self.flash_cooldown: int = obs['flash_cooldown']

    @property
    def can_flash(self) -> bool:
        return self.flash_cooldown == 0


class Monster(Character):
    def __init__(self, obs: dict):
        super(Monster, self).__init__(obs)
        self.monster_interval: int = obs['monster_interval']
        self.speed: int = obs['speed']
        self.is_in_view: bool = bool(obs['is_in_view'])


class Organ(Character):
    def __init__(self, obs: dict):
        super(Organ, self).__init__(obs)
        # treasure的config_id是从0开始的连续整数编号
        # buff的config_id 是接着 treasure 的编号继续排列
        self.status: bool = bool(obs['status'])
        """
        1=可获取, 0=不可获取, 
        实际上被收集后整个organ会直接从观测中消失, 即无法访问到status=0的情况
        """
        self.sub_type: int = obs['sub_type']
        """1=宝箱, 2=加速 buff"""
        # ========== custom
        self.cooldown: int = 0


class RawObs:
    def __init__(self, env_obs: dict):
        # ==========
        frame_state: dict = env_obs['frame_state']
        env_info: dict = env_obs['env_info']
        # ========== current env
        self.step: int = env_obs['step_no']
        """env.reset()后的step_no=0"""
        self.legal_action: list[bool] = env_obs['legal_action']
        self.map_view: np.ndarray = np.array(env_obs['map_info'])
        self.hero: Hero = Hero(frame_state['heroes'])
        self.monsters: list[Monster] = [Monster(d) for d in frame_state['monsters']]
        self.treasures: list[Organ] = [Organ(d) for d in frame_state['organs'] if d['sub_type'] == 1]
        self.buffs: list[Organ] = [Organ(d) for d in frame_state['organs'] if d['sub_type'] == 2]
        self.treasure_id: list[int] = env_info['treasure_id']
        """该变量记录的是本局内**没有被收集**的宝箱序号, 并非所有"""
        # ========== statistic
        self.collected_buff: int = env_info['collected_buff']
        self.flash_count: int = env_info['flash_count']
        self.step_score: float = env_info['step_score']
        self.total_score: float = env_info['total_score']
        self.treasure_score: float = env_info['treasure_score']
        self.treasures_collected: int = env_info['treasures_collected']
        # ========== env setting
        self.buff_refresh_time: int = env_info['buff_refresh_time']
        self.flash_cooldown_max: int = env_info['flash_cooldown_max']
        self.max_step: int = env_info['max_step']
        self.monster_init_speed: int = env_info['monster_init_speed']
        self.monster_interval: int = env_info['monster_interval']
        self.monster_speed_boost_step: int = env_info['monster_speed_boost_step']
        self.total_buff: int = env_info['total_buff']
        self.total_treasure: int = env_info['total_treasure']
        self.total_buff: int = env_info['total_buff']
        # ========== 忽略以下冗余字段, 已经在别处出现过了
        # env_info['pos']
        # env_info['step_no]
        # heroes[]['treasure_score']
        # heroes[]['step_score']
        # heroes[]['treasure_collected_count']


class FullObs(RawObs):
    def __init__(self, env_obs: dict):
        super().__init__(env_obs)
        # ========== hero
        self.hero_last: Hero | None = None
        self.hero_speed: int = 1
        self.action_preferred = [1] * len(self.legal_action)
        """过滤撞墙和不可用动作"""
        self.pos_history: deque[tuple[int, int]] = deque(maxlen=10)
        """记录历史10步坐标(不包括当前步), 用于怪物2的生成点判断"""
        # ========== map
        self.map_full: np.ndarray = np.full((int(MAP_SIZE), int(MAP_SIZE)), -1)
        """-1表示未探索, 0或1表示已探索"""
        self.map_explore_rate: float = 0.
        self.map_new_discover: int = 0
        # ========== organ
        self.treasure_full: list[Organ | None] = [None] * self.total_treasure
        self.buff_full: dict[int, Organ | None] = {}
        for i in range(self.total_treasure, self.total_treasure + self.total_buff):
            self.buff_full[i] = None
        # ========== score
        self.

        self.update(self)

    def update(self, obs: RawObs):
        # !!! self.hero is old before update !!!
        self.update_info(obs)
        # attributes in self is new now

        # update preferred action
        direction_index = [5, 2, 1, 0, 3, 6, 7, 8]
        map_slice = self.map_view[9:12, 9:12].reshape(-1)
        self.action_preferred = self.legal_action.copy()
        for i, idx in enumerate(direction_index):
            self.action_preferred[i] = map_slice[idx]
            
        # update trajectory
        if self.step > 0:
            self.pos_history.append((self.hero_last.x, self.hero_last.z))

        # update full map
        unknown_count_old = np.sum(self.map_full == -1)
        self.update_map(self.hero.x, self.hero.z, obs.map_view)
        unknown_count = np.sum(self.map_full == -1)
        self.map_explore_rate = 1 - (unknown_count / MAP_SIZE ** 2)
        self.map_new_discover = unknown_count_old - unknown_count

        # update organ
        self.update_organ(obs)


    def update_info(self, obs: RawObs):
        # TODO 部分增量没有记录
        self.step = obs.step
        self.legal_action = obs.legal_action
        self.map_view = obs.map_view
        self.hero_last = self.hero
        self.hero = obs.hero
        self.monsters = obs.monsters
        self.treasures = obs.treasures
        self.buffs = obs.buffs
        # ========== statistic
        self.collected_buff = obs.collected_buff
        self.flash_count = obs.flash_count
        self.step_score = obs.step_score
        self.total_score = obs.total_score
        self.treasure_score = obs.treasure_score
        self.treasures_collected = obs.treasures_collected
        # ========== don't need to update env setting
        # ========== 
        self.hero_speed = 2 if self.hero.buff_remaining_time > 0 else 1



    def update_map(self, x: int, z: int, map_view: np.ndarray):
        view_size = 21
        half_size = view_size // 2

        x_min = x - half_size
        x_max = x + half_size + 1
        z_min = z - half_size
        z_max = z + half_size + 1

        global_x_start = max(0, x_min)
        global_x_end = min(128, x_max)
        global_z_start = max(0, z_min)
        global_z_end = min(128, z_max)

        view_x_start = max(0, -x_min)
        view_x_end = view_size - max(0, x_max - 128)
        view_z_start = max(0, -z_min)
        view_z_end = view_size - max(0, z_max - 128)

        self.map_full[global_z_start:global_z_end, global_x_start:global_x_end] = \
            map_view[view_z_start:view_z_end, view_x_start:view_x_end]

    def update_organ(self, obs: RawObs):
        # add new seen treasures
        for o in obs.treasures:
            if self.treasure_full[o.id] is None:
                self.treasure_full[o.id] = o
        # mark collected treasures
        for o in self.treasure_full:
            if o and o.id not in obs.treasure_id:
                o.status = 0
        # add new seen buffs
        for o in obs.buffs:
            if self.buff_full[o.id] is None:
                self.buff_full[o.id] = o
        # refresh buffs cooldown
        for k, o in self.buff_full.items():
            if o is None:
                continue
            condition = [
                is_pos_neighbor(self.hero.x, self.hero.z, o.x, o.z),
                self.hero.buff_remaining_time == 49
            ]
            if all(condition):
                o.cooldown = self.buff_refresh_time
            # count down
            o.cooldown = max(o.cooldown - 1, 0)
