-- ============================================================================
-- 峡谷追猎 (Gorge Chase) - 强化学习训练场景
-- 路径: scripts/gameplay/gorge_chase/main.lua
-- ============================================================================
-- 需求规格: specs/infinity_valley/requirements_v2.md
-- 场景类型: PVE 单智能体强化学习
-- 地图大小: 128x128 栅格
-- 用途: 2026腾讯开悟公开赛（服创大赛D类）初赛
-- ============================================================================

-- ============================================================================
-- 加载模块
-- ============================================================================
-- 注意：运行时 Lua 环境是沙箱，dofile/loadfile 被禁用，请使用 require
require("skills.flash_skill")

-- 加载控制系统
local Control = require("control.control")
local GridUnit = require("ecs.grid_unit")
local ECS = require("ecs.ecs")
local ECSCommon = require("ecs.common")
local GorgeEnemyElement = require("ecs.elements.gorge_enemy_spawn")
local GorgePlayerElement = require("ecs.elements.xiaowu_player")
local RobotProfiles = require("ecs.robot_profiles")
local GorgeChaseFrame = require("gameplay.gorge_chase.frame")
local GorgeChaseStatus = require("gameplay.gorge_chase.status")
local GorgeChaseWorld = require("gameplay.gorge_chase.world")


-- ============================================================================
-- 默认配置（Default Configuration）
-- ============================================================================
local function load_default_config_from_env()
    -- 由 Rust 侧将 config/gorge_chase/env_config.toml 注入到 Lua 全局
    local injected = _G.__gorge_chase_user_config
    if type(injected) == "table" then
        return ECSCommon.deep_copy(injected)
    end
    return {}
end

local DEFAULT_CONFIG = load_default_config_from_env()

-- ============================================================================
-- 全局状态管理
-- ============================================================================
local game_state = {
    -- 实体引用
    player = nil,                -- 智能体（鲁班七号）
    enemies = {},                -- 野怪列表（神秘魔种，最多2个）
    treasures = {},              -- 宝箱列表
    buffs = {},                  -- 加速buff列表
    
    -- 任务状态
    current_step = 0,            -- 当前步数
    max_step = 1000,             -- 最大步数
    task_status = "running",     -- 任务状态: running/completed/failed
    
    -- 计分数据
    score = 0,                   -- 总得分
    step_score = 0,              -- 步数得分
    treasure_score = 0,          -- 宝箱得分
    treasures_collected = 0,     -- 已收集宝箱数
    total_treasures = 0,         -- 总宝箱数量
    skill_usage_count = 0,
    buff_obtained_count = 0,
    
    -- 智能体状态

    player_has_speed_buff = false,    -- 是否拥有加速buff
    player_speed_buff_remaining = 0,  -- 加速buff剩余时间（步数）
    player_flash_cooldown = 0,        -- 闪现冷却剩余时间（步数）
    player_move_bonus = 0,
    
    -- 怪物生成控制
    second_enemy_spawned = false,     -- 第二个怪物是否已生成
    second_enemy_spawn_step = 300,    -- 第二个怪物生成步数
    monster_speed_boosted = false,    -- 怪物是否已加速
    monster_speed_boost_step = 500,   -- 怪物加速增益出现步数
    
    -- buff刷新控制
    buff_refresh_counters = {},       -- buff刷新计数器 {buff_id: remaining_steps}
    
    -- 地图信息
    map_width = 128,
    map_height = 128,
    current_map_id = 1,
    
    -- 移动状态（单帧同步）
    player_move_path = nil,           -- 实际移动路径（用于收集物品）
    monster_speed_bonus = 0,
}

-- 怪物AI状态
local enemy_states = {}

-- 用户配置（从 reset 传入）
local usr_config = {}

-- 系统参数统一走 usr_config（由 env_config.toml / reset(user_config) 注入）

-- 方向向量（与 0-7 方向编号一致）
local DIRECTION_VECTORS = {
    [0] = {x = 1,  z = 0},   -- RIGHT
    [1] = {x = 1,  z = -1},  -- UP_RIGHT
    [2] = {x = 0,  z = -1},  -- UP
    [3] = {x = -1, z = -1},  -- UP_LEFT
    [4] = {x = -1, z = 0},   -- LEFT
    [5] = {x = -1, z = 1},   -- DOWN_LEFT
    [6] = {x = 0,  z = 1},   -- DOWN
    [7] = {x = 1,  z = 1},   -- DOWN_RIGHT
}
local merge_tags = ECSCommon.merge_tags


-- ============================================================================
-- 技能系统
-- ============================================================================

--- 初始化技能
local function init_skills()
    -- 清空旧技能
    skill.clear()
    
    -- 注册闪现技能（使用技能脚本）
    register_flash_skill({
        cooldown = usr_config.flash_cooldown,
        distance = usr_config.flash_distance,
        distance_diagonal = usr_config.flash_distance_diagonal,
    })
end


--- 技能使用回调（由UI触发）
function on_skill_used(skill_id)
    log("[Skill] Skill used from UI: " .. skill_id)
    
    if skill_id == "flash" then
        -- 检查技能是否就绪
        local flash_skill = skill.get("flash")
        if not flash_skill then
            log_error("[Skill] Flash skill not found")
            return
        end
        
        if flash_skill.cooldown > 0 then
            log_warn(string.format("[Skill] Flash is on cooldown: %d steps remaining", flash_skill.cooldown))
            return
        end
        
        if not flash_skill.enabled then
            log_warn("[Skill] Flash is disabled")
            return
        end
        
        -- 触发闪现（这里可以添加特殊的闪现效果）
        log("[Skill] Flash skill activated!")
        -- 注意：实际的闪现移动已经通过execute_agent_action处理
    end
end

-- ============================================================================
-- 工具函数
-- ============================================================================

--- 深拷贝表
local deep_copy = ECSCommon.deep_copy

--- 合并配置（usr_config 覆盖 default_config）
local merge_config = ECSCommon.merge_config
local clamp = ECSCommon.clamp
local collect_sorted_keys = ECSCommon.collect_sorted_keys
local key_diff_report = ECSCommon.key_diff_report

--- 计算 hero->target 的 L2 距离桶编号（0..5）
--- 128x128 地图最大对角线距离约为 181，均匀划分为6个桶，每个桶约30格
--- @param hero_pos table|nil {x:int, z:int}
--- @param target_pos table|nil {x:int, z:int}
--- @return number bucket 0..5
local function calc_hero_l2_distance_bucket(hero_pos, target_pos)
    if not hero_pos or not target_pos then
        return 0
    end

    local hx = hero_pos.x or 0
    local hz = hero_pos.z or 0
    local tx = target_pos.x or 0
    local tz = target_pos.z or 0

    local dx = tx - hx
    local dz = tz - hz

    local d = math.sqrt(dx * dx + dz * dz)
    -- 128x128地图最大距离 = sqrt(127^2 + 127^2) ≈ 179.6，取180
    -- 均匀划分为6个桶：[0,30) -> 0, [30,60) -> 1, ..., [150,180] -> 5
    local max_distance = 180
    local bucket = math.floor(d * 6 / max_distance)
    if bucket < 0 then bucket = 0 end
    if bucket > 5 then bucket = 5 end

    return bucket
end

--- 计算 hero->target 的相对方向编号（0..8）
--- 0=不可达(兜底)
--- 1=东, 2=东北, 3=北, 4=西北, 5=西, 6=西南, 7=南, 8=东南
--- 说明：z 轴与脚本内部 y 轴一致（北=dz<0, 南=dz>0）
--- @param hero_pos table|nil {x:int, z:int}
--- @param target_pos table|nil {x:int, z:int}
--- @return number dir_id 0..8
local function calc_hero_relative_direction_id(hero_pos, target_pos)
    if not hero_pos or not target_pos then
        return 0
    end

    local hx = hero_pos.x or 0
    local hz = hero_pos.z or 0
    local tx = target_pos.x or 0
    local tz = target_pos.z or 0

    local dx = tx - hx
    local dz = tz - hz

    if dx == 0 and dz == 0 then
        return 0
    end

    local sx = (dx > 0 and 1) or (dx < 0 and -1) or 0
    local sz = (dz > 0 and 1) or (dz < 0 and -1) or 0

    if sx == 1 and sz == 0 then return 1 end     -- 东
    if sx == 1 and sz == -1 then return 2 end    -- 东北
    if sx == 0 and sz == -1 then return 3 end    -- 北
    if sx == -1 and sz == -1 then return 4 end   -- 西北
    if sx == -1 and sz == 0 then return 5 end    -- 西
    if sx == -1 and sz == 1 then return 6 end    -- 西南
    if sx == 0 and sz == 1 then return 7 end     -- 南
    if sx == 1 and sz == 1 then return 8 end     -- 东南

    return 0
end

--- 检查坐标是否在地图内
local function is_in_bounds(x, z)
    return ECSCommon.is_in_bounds(x, z, game_state.map_width, game_state.map_height)
end


--- 检查坐标是否可通行
local function is_walkable(x, z)
    if not is_in_bounds(x, z) then
        return false
    end
    return map.is_walkable(x, z)
end

--- 检查斜向移动是否可行（考虑相邻边的阻挡情况）
--- @param from_x number 起始x坐标
--- @param from_z number 起始z坐标
--- @param to_x number 目标x坐标
--- @param to_z number 目标z坐标
--- @return boolean 是否可以移动
local function can_move_diagonally(from_x, from_z, to_x, to_z)
    return ECSCommon.can_move_diagonally(from_x, from_z, to_x, to_z, is_walkable)
end

--- 随机选择
local random_choice = ECSCommon.random_choice

--- 生成随机整数 [min, max]
local function random_int(min, max)
    return math.random(min, max)
end

local function to_grid_pos(x, z, extra)
    local p = extra or {}
    p.x = x
    p.z = z
    return p
end

local function get_z(pos)
    if not pos then
        return nil
    end

    local z = tonumber(pos.z)
    if z ~= nil then
        return z
    end

    local y = tonumber(pos.y)
    if y ~= nil then
        return y
    end

    local p = pos.position
    if type(p) == "table" then
        return tonumber(p.z) or tonumber(p[2]) or tonumber(p.y)
    end

    return nil
end

-- ============================================================================
-- 位置历史（用于：第二只怪物按“玩家10步前位置”生成）
-- ============================================================================

local function push_player_pos_history()
    if not game_state or not game_state.player then
        return
    end

    if not game_state.player_pos_history then
        game_state.player_pos_history = {}
    end

    local gx = math.floor(game_state.player.x)
    local pz = get_z(game_state.player)
    if pz == nil then
        return
    end
    local gz = math.floor(pz)

    table.insert(game_state.player_pos_history, {
        step = game_state.current_step or 0,
        x = gx,
        z = gz,
    })

    -- 只保留最近一段（足够覆盖 10 步回溯）
    while #game_state.player_pos_history > 32 do
        table.remove(game_state.player_pos_history, 1)
    end
end

local function get_player_pos_steps_ago(n)
    local history = game_state and game_state.player_pos_history or nil
    if not history or #history == 0 then
        return nil
    end

    local cur = game_state.current_step or 0
    local target_step = cur - (n or 0)

    for i = #history, 1, -1 do
        local h = history[i]
        if h and h.step == target_step then
            return to_grid_pos(h.x, get_z(h))
        end
    end

    -- 找不到精确步号时，退化为“最早的可用位置”
    local h0 = history[1]
    if h0 then
        return to_grid_pos(h0.x, get_z(h0))
    end

    return nil
end

local function find_nearest_walkable_cell(gx, gz, max_radius)
    if is_walkable(gx, gz) then
        return to_grid_pos(gx, gz)
    end

    local rmax = max_radius or 10
    for r = 1, rmax do
        local found = nil
        GridUnit.for_each_square_radius(gx, gz, r, function(x, z)
            if not found and is_walkable(x, z) then
                found = to_grid_pos(x, z)
            end
        end)
        if found then
            return found
        end
    end

    return nil
end

local function get_enemy_id_at_grid(gx, gz, exclude_id)
    local enemies = entity.get_all_enemies()
    for _, e in ipairs(enemies or {}) do
        if (exclude_id == nil or e.id ~= exclude_id)
            and math.floor(e.x) == gx
                and math.floor(e.z) == gz then
            return e.id
        end
    end
    return nil
end

local function find_nearest_walkable_unoccupied_cell(gx, gz, max_radius, exclude_id)
    local occ = get_enemy_id_at_grid(gx, gz, exclude_id)
    if is_walkable(gx, gz) and not occ then
        return to_grid_pos(gx, gz)
    end

    local rmax = max_radius or 10
    for r = 1, rmax do
        local found = nil
        GridUnit.for_each_square_radius(gx, gz, r, function(x, z)
            if not found and is_walkable(x, z) and not get_enemy_id_at_grid(x, z, exclude_id) then
                found = to_grid_pos(x, z)
            end
        end)
        if found then
            return found
        end
    end

    return nil
end

-- ============================================================================
-- 配置处理
-- ============================================================================


--- 验证并修正配置（确保值在合法范围内，负数或nil当作0处理）
local function validate_config(config)
    -- 修正宝箱数量（负数或nil当作0，超过10限制为10）
    if not config.treasure_count or config.treasure_count < 0 then
        config.treasure_count = 0
    elseif config.treasure_count > 10 then
        config.treasure_count = 10
    end
    
    -- 修正buff数量（负数或nil当作0，超过2限制为2）
    if not config.buff_count or config.buff_count < 0 then
        config.buff_count = 0
    elseif config.buff_count > 2 then
        config.buff_count = 2
    end
    
    -- 修正buff最小间距（默认10，范围[0, 100]）
    if not config.buff_min_distance or config.buff_min_distance < 0 then
        config.buff_min_distance = 10
    elseif config.buff_min_distance > 100 then
        config.buff_min_distance = 100
    end
    
    -- 修正怪物数量（nil使用默认2；负数当作0；超过2限制为2）
    if config.monster_count == nil then
        config.monster_count = 2
    elseif config.monster_count < 0 then
        config.monster_count = 0
    elseif config.monster_count > 2 then
        config.monster_count = 2
    end
    
    -- 修正怪物速度（nil使用默认1；负数当作0；超过4限制为4）
    if config.monster_speed == nil then
        config.monster_speed = 1
    elseif config.monster_speed < 0 then
        config.monster_speed = 0
    elseif config.monster_speed > 4 then
        config.monster_speed = 4
    end
    
    -- 修正buff额外速度（负数或nil当作1，超过10限制为10）
    if not config.buff_extra_speed or config.buff_extra_speed < 1 then
        config.buff_extra_speed = 1
    elseif config.buff_extra_speed > 10 then
        config.buff_extra_speed = 10
    end
    
    -- 修正最大步数（负数或nil当作1，超过2000限制为2000）
    if not config.max_step or config.max_step < 1 then
        config.max_step = 1
    elseif config.max_step > 2000 then
        config.max_step = 2000
    end
    
    -- 修正其他配置项（确保存在）
    if config.map_random == nil then
        config.map_random = true
    end
    
    if not config.map_id then
        config.map_id = 1
    end
    
    if not config.buff_refresh_time then
        config.buff_refresh_time = 200
    end
    
    if not config.buff_duration then
        config.buff_duration = 50
    end
    
    if not config.monster_interval then
        config.monster_interval = 300
    end
    
    if not config.monster_speed_boost_step then
        config.monster_speed_boost_step = 500
    end
    
    if not config.flash_cooldown then
        config.flash_cooldown = 100
    end
    
    if not config.flash_distance then
        config.flash_distance = 10
    end
    
    if not config.flash_distance_diagonal then
        config.flash_distance_diagonal = 8
    end

    -- 修正系统配置项（可由 env_config.toml 的 [environment] 注入）
    if config.player_speed_normal == nil then
        config.player_speed_normal = 1
    elseif config.player_speed_normal < 0 then
        config.player_speed_normal = 0
    elseif config.player_speed_normal > 4 then
        config.player_speed_normal = 4
    end

    if config.treasure_value == nil then
        config.treasure_value = 100
    elseif config.treasure_value < 0 then
        config.treasure_value = 0
    end

    if config.step_score_multiplier == nil then
        config.step_score_multiplier = 1.5
    elseif config.step_score_multiplier < 0 then
        config.step_score_multiplier = 0
    end

    -- 视野固定为 21x21：半径必须为 10（2*10+1=21）
    local vr = tonumber(config.vision_range)
    if vr == nil then
        config.vision_range = 10
    else
        config.vision_range = math.floor(vr + 1e-6)
        if config.vision_range ~= 10 then
            log_warn(string.format(
                "[Config] vision_range=%s ignored, forced to 10 for 21x21 vision",
                tostring(vr)
            ))
            config.vision_range = 10
        end
    end
    
    log("[Config] Configuration validated and corrected")
    return true
end

-- ============================================================================
-- 实体生成
-- ============================================================================

--- 获取敌人出生点（从地图对象中读取）
local function get_enemy_spawn_points()
    local raw_points = GorgeEnemyElement.query_spawn_points(ECS)
    local spawn_points = {}
    for _, p in ipairs(raw_points or {}) do
        table.insert(spawn_points, to_grid_pos(p.x, get_z(p), { obj = p.obj }))
    end
    
    -- 如果地图没有出生点，搜索可通行区域
    if #spawn_points == 0 then
        log_warn("[Spawn] No enemy spawn points in map, searching for walkable position")
        
        -- 尝试在地图的多个区域找到可通行位置
        local search_areas = {
            {x = game_state.map_width * 3 // 4, z = game_state.map_height * 3 // 4},  -- 右下
            {x = game_state.map_width * 1 // 4, z = game_state.map_height * 3 // 4},  -- 左下
            {x = game_state.map_width * 3 // 4, z = game_state.map_height * 1 // 4},  -- 右上
        }
        
        for _, area in ipairs(search_areas) do
            for radius = 0, 30 do
                local found = nil
                GridUnit.for_each_square_radius(area.x, area.z, radius, function(x, y)
                    if found then
                        return
                    end
                    if x >= 0 and x < game_state.map_width and 
                       y >= 0 and y < game_state.map_height and 
                       is_walkable(x, y) then
                        found = to_grid_pos(x, y)
                    end
                end)
                if found then
                    table.insert(spawn_points, found)
                    log(string.format("[Spawn] Using default enemy spawn at (%d, %d)", found.x, found.z))
                    return spawn_points
                end
            end
        end
        
        log_warn("[Spawn] No walkable position found for enemy spawn")
    else
        log(string.format("[Spawn] Found %d enemy spawn points", #spawn_points))
    end
    
    return spawn_points
end

    --- 在指定区域内搜索可通行点
    --- @param start_x number 搜索起始x
    --- @param start_z number 搜索起始z
    --- @param end_x number 搜索结束x
    --- @param end_z number 搜索结束z
    --- @param prefer_corner string 优先搜索的角落 "tl"/"tr"/"bl"/"br"
    --- @return table|nil 可通行点坐标 {x, z}
    local function find_walkable_in_region(start_x, start_z, end_x, end_z, prefer_corner)
        local map_w = game_state.map_width
        local map_h = game_state.map_height

        -- 确定搜索方向
        local step_x = (end_x >= start_x) and 1 or -1
        local step_z = (end_z >= start_z) and 1 or -1

        -- 根据优先角落调整搜索顺序
        local x_first = (prefer_corner == "tl" or prefer_corner == "bl")

        if x_first then
            -- 优先x方向
            for x = start_x, end_x, step_x do
                for z = start_z, end_z, step_z do
                    if x >= 0 and x < map_w and z >= 0 and z < map_h then
                        if is_walkable(x, z) then
                            return to_grid_pos(x, z)
                        end
                    end
                end
            end
        else
            -- 优先y方向
            for z = start_z, end_z, step_z do
                for x = start_x, end_x, step_x do
                    if x >= 0 and x < map_w and z >= 0 and z < map_h then
                        if is_walkable(x, z) then
                            return to_grid_pos(x, z)
                        end
                    end
                end
            end
        end

        return nil
    end

    --- 根据地图尺寸计算四个角落的出生点位置
    --- 策略：将地图分为四个象限，在对应象限中搜索离角落最近的可通行点
    --- @param corner number 角落编号（1:左上角，2:右上角，3:左下角，4:右下角）
    --- @return table|nil 出生点坐标 {x, z}
    local function calculate_corner_spawn_point(corner)
        local map_w = game_state.map_width
        local map_h = game_state.map_height
        local half_w = map_w // 2
        local half_h = map_h // 2

        -- 边距：距离地图边缘的格子数
        local margin = 3

        log(string.format("[Spawn] Calculating corner %d, map size: %dx%d", corner, map_w, map_h))

        local pos = nil

        if corner == 1 then
            -- 左上角：在左上象限搜索，从(margin,margin)向中心搜索
            pos = find_walkable_in_region(margin, margin, half_w, half_h, "tl")
        elseif corner == 2 then
            -- 右上角：在右上象限搜索，从(max-margin,margin)向中心搜索
            pos = find_walkable_in_region(map_w - 1 - margin, margin, half_w, half_h, "tr")
        elseif corner == 3 then
            -- 左下角：在左下象限搜索，从(margin,max-margin)向中心搜索
            pos = find_walkable_in_region(margin, map_h - 1 - margin, half_w, half_h, "bl")
        elseif corner == 4 then
            -- 右下角：在右下象限搜索，从(max-margin,max-margin)向中心搜索
            pos = find_walkable_in_region(map_w - 1 - margin, map_h - 1 - margin, half_w, half_h, "br")
        end

        if pos then
            log(string.format("[Spawn] Found walkable position at (%d, %d) for corner %d", pos.x, pos.z, corner))
            return pos
        end

        -- 如果在对应象限没找到，扩展到整个地图搜索
        log_warn(string.format("[Spawn] Corner %d quadrant search failed, searching entire map", corner))

        -- 根据角落确定搜索起点和方向
        if corner == 1 then
            pos = find_walkable_in_region(0, 0, map_w - 1, map_h - 1, "tl")
        elseif corner == 2 then
            pos = find_walkable_in_region(map_w - 1, 0, 0, map_h - 1, "tr")
        elseif corner == 3 then
            pos = find_walkable_in_region(0, map_h - 1, map_w - 1, 0, "bl")
        elseif corner == 4 then
            pos = find_walkable_in_region(map_w - 1, map_h - 1, 0, 0, "br")
        end

        if pos then
            log(string.format("[Spawn] Found walkable position at (%d, %d) for corner %d (full search)", pos.x, pos.z, corner))
            return pos
        end

        log_warn(string.format("[Spawn] No walkable position found for corner %d", corner))
        return nil
    end

    --- 获取固定出生点（根据配置的角落编号计算）
    --- @param start_position number 角落编号（1:左上角，2:右上角，3:左下角，4:右下角）
    local function get_fixed_spawn_point(start_position)
        -- 验证参数范围
        local corner = start_position or 1
        if corner < 1 or corner > 4 then
            log_warn(string.format("[Spawn] Invalid start_position %d, using default 1", corner))
            corner = 1
        end

        log(string.format("[Spawn] Fixed spawn: corner %d", corner))
        return calculate_corner_spawn_point(corner)
    end

    --- 获取“玩家视野最外围一层”的候选出生点（Chebyshev 距离 = vision_range）
    --- @param cx number 玩家格子x
    --- @param cz number 玩家格子z
    --- @param vr number 视野半径（21x21 时为 10）
    --- @return table 候选点数组 { {x, z}, ... }
    local function collect_vision_outer_ring_points(cx, cz, vr)
        local candidates = {}
        for dy = -vr, vr do
            for dx = -vr, vr do
                if math.max(math.abs(dx), math.abs(dy)) == vr then
                    local x = cx + dx
                    local z = cz + dy
                    if is_in_bounds(x, z) and is_walkable(x, z) then
                        table.insert(candidates, to_grid_pos(x, z))
                    end
                end
            end
        end
        return candidates
    end

    --- 怪物1出生规则：在玩家视野（21x21）的最外围一层随机出生，非固定位置
    --- @return table|nil 出生点坐标 {x, z}
    local function get_first_monster_spawn_on_vision_outer_ring()
        if not game_state.player then
            return nil
        end

        local cx = math.floor(game_state.player.x)
        local cz = math.floor(game_state.player.z)
        local vr = math.floor((usr_config.vision_range or 10) + 1e-6)

        local ring = collect_vision_outer_ring_points(cx, cz, vr)
        if #ring > 0 then
            local pos = random_choice(ring)
            if pos then
                log(string.format(
                    "[Spawn] Monster 1 choose vision outer-ring position (%d, %d), candidates=%d, vr=%d",
                    pos.x,
                    pos.z,
                    #ring,
                    vr
                ))
                return pos
            end
        end

        -- 兜底：若最外层全不可走，则向内层回退（仍优先靠外）
        for r = vr - 1, 1, -1 do
            local inner = collect_vision_outer_ring_points(cx, cz, r)
            if #inner > 0 then
                local pos = random_choice(inner)
                if pos then
                    log_warn(string.format(
                        "[Spawn] Monster 1 outer ring blocked, fallback to inner ring r=%d at (%d, %d), candidates=%d",
                        r,
                        pos.x,
                        pos.z,
                        #inner
                    ))
                    return pos
                end
            end
        end

        log_warn("[Spawn] Monster 1 cannot find walkable cell in player's vision rings")
        return nil
    end






--- 获取怪物出生点（从地图对象中读取）
local function get_monster_spawn_point(monster_id)
    -- 怪物1：按需求从“玩家视野最外围一层”随机选择，非固定位置
    if monster_id == 1 then
        local pos = get_first_monster_spawn_on_vision_outer_ring()
        if pos then
            return pos
        end
        -- 如果极端地图导致视野圈无可走点，再回退到旧逻辑
        log_warn("[Spawn] Monster 1 fallback to legacy spawn-point logic")
    end

    -- 从地图获取敌人出生点
    local spawn_points = get_enemy_spawn_points()
    
    -- 如果没有出生点，返回nil
    if #spawn_points == 0 then
        log_warn("[Spawn] No enemy spawn points available")
        return nil
    end
    
    -- 使用 monster_id 作为索引选择出生点（循环使用）
    local index = ((monster_id - 1) % #spawn_points) + 1
    local pos = spawn_points[index]
    
    -- 确保位置可通行
    if is_walkable(pos.x, pos.z) then
        return pos
    end
    
    -- 如果不可通行，搜索附近可通行点
    for radius = 1, 10 do
        local found = nil
        GridUnit.for_each_square_radius(pos.x, pos.z, radius, function(x, y)
            if not found and is_walkable(x, y) then
                found = to_grid_pos(x, y)
            end
        end)
        if found then
            log(string.format("[Spawn] Found walkable position near spawn point: (%d, %d)", found.x, found.z))
            return found
        end
    end
    
    log_warn("[Spawn] Cannot find walkable position for monster spawn")
    return nil
end

--- 获取智能体随机出生点（从所有可通行位置中随机选择，排除宝箱和加速buff位置）
local function get_random_spawn_point()
    local map_w = game_state.map_width
    local map_h = game_state.map_height

    -- 构建需要排除的位置集合（宝箱和buff所在的格子）
    local excluded_positions = {}

    -- 排除宝箱位置
    for _, treasure in ipairs(game_state.treasures or {}) do
        local grid_x = math.floor(treasure.x)
        local grid_z = math.floor(treasure.z)
        excluded_positions[grid_x .. "," .. grid_z] = true
    end

    -- 排除加速buff位置
    for _, buff in ipairs(game_state.buffs or {}) do
        local grid_x = math.floor(buff.x)
        local grid_z = math.floor(buff.z)
        excluded_positions[grid_x .. "," .. grid_z] = true
    end

    -- 收集可通行位置
    local walkable_positions = {}
    for z = 0, map_h - 1 do
        for x = 0, map_w - 1 do
            local key = x .. "," .. z
            if is_walkable(x, z) and not excluded_positions[key] then
                table.insert(walkable_positions, to_grid_pos(x, z))
            end
        end
    end

    if #walkable_positions == 0 then
        log_warn("[Spawn] No walkable positions found for random player spawn")
        return nil
    end

    local selected = random_choice(walkable_positions)
    if selected then
        log(string.format("[Spawn] Random walkable player spawn at (%d, %d), candidates=%d", selected.x, selected.z, #walkable_positions))
    end
    return selected
end

--- 判断玩家出生格是否被宝箱/buff占用（避免 entity.spawn 失败）
local function is_player_spawn_cell_occupied(gx, gz)
    for _, treasure in ipairs(game_state.treasures or {}) do
        if math.floor(treasure.x) == gx and math.floor(treasure.z) == gz then
            return true, "treasure"
        end
    end

    for _, buff in ipairs(game_state.buffs or {}) do
        if math.floor(buff.x) == gx and math.floor(buff.z) == gz then
            return true, "buff"
        end
    end

    return false, ""
end

--- 生成智能体
local function spawn_player()
    local spawn_pos = get_random_spawn_point()

    -- 小悟出生逻辑：
    -- - start_random=true：从地图中的小悟出生点随机选择
    -- - start_random=false：优先按小悟出生点序号选择；无点时回退到固定角落
    local map_spawn_points = GorgePlayerElement.query_spawn_points(ECS)
    if usr_config.start_random then
        if #map_spawn_points > 0 then
            spawn_pos = random_choice(map_spawn_points)
            if spawn_pos then
                spawn_pos = to_grid_pos(spawn_pos.x, get_z(spawn_pos), { obj = spawn_pos.obj })
                log(string.format("[Spawn] Randomly selected gorge player spawn at (%d, %d)", spawn_pos.x, spawn_pos.z))
            end
        else
            log_warn("[Spawn] start_random=true but no gorge player spawn found (spawn_point_player), fallback to random walkable spawn")
            spawn_pos = get_random_spawn_point()
            if not spawn_pos then
                log_warn("[Spawn] Random walkable spawn failed, fallback to fixed corner spawn")
                spawn_pos = get_fixed_spawn_point(usr_config.start)
            end
        end
    else
        if #map_spawn_points > 0 then
            local idx = clamp(math.floor((tonumber(usr_config.start) or 1) + 1e-6), 1, #map_spawn_points)
            spawn_pos = map_spawn_points[idx]
            if spawn_pos then
                spawn_pos = to_grid_pos(spawn_pos.x, get_z(spawn_pos), { obj = spawn_pos.obj })
                log(string.format("[Spawn] Using gorge player spawn from map at (%d, %d)", spawn_pos.x, spawn_pos.z))
            end
        else
            spawn_pos = get_fixed_spawn_point(usr_config.start)
        end
    end

    -- 兜底：若出生点不可通行，先在附近搜索，再回退到固定角落
    if spawn_pos and not is_walkable(spawn_pos.x, spawn_pos.z) then
        local found = nil
        for radius = 1, 10 do
            GridUnit.for_each_square_radius(spawn_pos.x, spawn_pos.z, radius, function(x, y)
                if not found and is_walkable(x, y) then
                    found = to_grid_pos(x, y)
                end
            end)
            if found then
                break
            end
        end
        if found then
            log_warn(string.format("[Spawn] Player spawn (%d, %d) not walkable, use nearby (%d, %d)", spawn_pos.x, spawn_pos.z, found.x, found.z))
            spawn_pos = found
        else
            log_warn("[Spawn] Player spawn not walkable and nearby search failed, fallback to random walkable spawn")
            spawn_pos = get_random_spawn_point()
            if not spawn_pos then
                log_warn("[Spawn] Random walkable spawn failed, fallback to fixed corner spawn")
                spawn_pos = get_fixed_spawn_point(usr_config.start)
            end
        end
    end
    
    if not spawn_pos then
        log_error("[Spawn] Failed to get player spawn position")
        game_state.task_status = "failed"
        return false
    end

    -- 若出生格被收集物占用，优先改为随机可通行点
    do
        local occupied, occupied_type = is_player_spawn_cell_occupied(spawn_pos.x, spawn_pos.z)
        if occupied then
            log_warn(string.format("[Spawn] Candidate player spawn (%d,%d) occupied by %s, fallback to random walkable spawn", spawn_pos.x, spawn_pos.z, occupied_type))
            local alt = get_random_spawn_point()
            if alt then
                spawn_pos = alt
            end
        end
    end
    
    log(string.format("[Spawn] Player at (%d, %d)", spawn_pos.x, spawn_pos.z))

    local player_profile = RobotProfiles.get("gorge_chase", "player") or {}
    local player_collision_radius = tonumber(player_profile.collision_size and player_profile.collision_size.radius) or 0.4
    
    -- 使用新API: 传入tags数组
    -- 栅格移动：实体位于格子中心 (x+0.5, y+0.5)
    local function try_spawn_player_at(pos)
        if not pos then
            return nil
        end
        return entity.spawn(player_profile.tag or "player", pos.x + 0.5, pos.z + 0.5, merge_tags({"player", "agent"}, player_profile.tags), {
            collision = {
                enabled = true,
                shape = { type = "circle", radius = player_collision_radius },
            },
            properties = {
                speed = usr_config.player_speed_normal,
            },
            visual = {
                display_name = "小悟",
                color = player_profile.color or {0, 255, 0, 255},
                visible = true,
            },
        })
    end

    local player_id = try_spawn_player_at(spawn_pos)

    -- 若首次创建失败，尝试备用出生点
    if not player_id then
        log_warn(string.format("[Spawn] First spawn attempt failed at (%d, %d), trying fallback candidates", spawn_pos.x, spawn_pos.z))
        local fallback_candidates = {
            get_random_spawn_point(),
            get_fixed_spawn_point(usr_config.start),
        }
        for _, candidate in ipairs(fallback_candidates) do
            if candidate then
                local occupied = is_player_spawn_cell_occupied(candidate.x, candidate.z)
                if not occupied then
                    player_id = try_spawn_player_at(candidate)
                    if player_id then
                        spawn_pos = candidate
                        break
                    end
                end
            end
        end
    end
    
    if not player_id then
        log_error("[Spawn] Failed to create player entity")
        game_state.task_status = "failed"
        return false
    end

    -- 若起点来自“小悟实例”物体，生成后移除源地图物体，避免额外方块占位
    GorgePlayerElement.consume_source(spawn_pos)

    -- 获取实体对象
    game_state.player = entity.get_player()
    game_state.player_move_bonus = clamp(math.floor((tonumber(player_profile.player_move_bonus) or 0) + 1e-6), 0, 3)
    if game_state.player and game_state.player.id then
        entity.set_property(game_state.player.id, "robot_profile", player_profile.class_name or "standard")
        entity.set_property(game_state.player.id, "player_move_bonus", game_state.player_move_bonus)
    end
    
    return true
end

--- 生成怪物
--- @param monster_id number
--- @param spawn_pos_override table|nil 可选：强制出生点 {x=grid_x, z=grid_z}
local function spawn_monster(monster_id, spawn_pos_override)
    -- 如果怪物速度为0，不生成怪物
    if usr_config.monster_speed <= 0 then
        log(string.format("[Spawn] Monster speed is 0, skipping monster %d", monster_id))
        return nil
    end

    local spawn_pos = spawn_pos_override or get_monster_spawn_point(monster_id)
    if not spawn_pos then
        log_warn(string.format("[Spawn] No spawn point for monster %d, skipping", monster_id))
        return nil
    end

    -- 兜底：若指定出生点不可走，找附近可走点
    if not is_walkable(spawn_pos.x, spawn_pos.z) then
        local fallback = find_nearest_walkable_cell(spawn_pos.x, spawn_pos.z, 10)
        if not fallback then
            log_warn(string.format("[Spawn] Monster %d spawn blocked at (%d,%d), and no fallback found", monster_id, spawn_pos.x, spawn_pos.z))
            return nil
        end
        spawn_pos = fallback
    end

    -- 防止多怪出生在同一格（例如怪物2在怪物1当前位置生成）
    local occupied_by = get_enemy_id_at_grid(spawn_pos.x, spawn_pos.z, nil)
    if occupied_by then
        local fallback = find_nearest_walkable_unoccupied_cell(spawn_pos.x, spawn_pos.z, 10, nil)
        if not fallback then
            log_warn(string.format(
                "[Spawn] Monster %d spawn occupied by monster %d at (%d,%d), and no unoccupied fallback found",
                monster_id,
                occupied_by,
                spawn_pos.x,
                spawn_pos.z
            ))
            return nil
        end
        log(string.format(
            "[Spawn] Monster %d spawn occupied by monster %d, relocate to (%d,%d)",
            monster_id,
            occupied_by,
            fallback.x,
            fallback.z
        ))
        spawn_pos = fallback
    end

    log(string.format("[Spawn] Monster %d at (%d, %d)", monster_id, spawn_pos.x, spawn_pos.z))

    local monster_profile = RobotProfiles.get("gorge_chase", "monster") or {}
    local speed_bonus = clamp(math.floor((tonumber(monster_profile.monster_speed_bonus) or 0) + 1e-6), 0, 3)
    local monster_speed = clamp((usr_config.monster_speed or 0) + speed_bonus, 0, 4)
    
    -- 如果全局加速已触发，出生时速度+1
    local is_boosted = game_state.monster_speed_boosted or false
    if is_boosted then
        monster_speed = monster_speed + 1
        log(string.format("[Spawn] Monster %d born with speed boost (speed: %d)", monster_id, monster_speed))
    end

    -- 使用新API: 传入tags数组
    local monster_entity_id = entity.spawn(
        (monster_profile.tag or "enemy") .. "_" .. monster_id,
        spawn_pos.x + 0.5,
        spawn_pos.z + 0.5,
        merge_tags({"enemy", "monster"}, monster_profile.tags),
        {
            collision = {
                enabled = true,
                shape = {
                    type = "circle",
                    radius = tonumber(monster_profile.collision_size and monster_profile.collision_size.radius) or 0.4,
                },
            },
            properties = {
                speed = monster_speed,
            },
            visual = {
                display_name = monster_profile.name or ("怪物" .. tostring(monster_id)),
                color = monster_profile.color or {255, 60, 60, 255},
                visible = true,
            },
        }
    )

    if not monster_entity_id then
        log_error(string.format("[Spawn] Failed to create monster %d entity", monster_id))
        return nil
    end

    GorgeEnemyElement.consume_source(spawn_pos)

    -- 初始化怪物状态（使用ID作为key）
    enemy_states[monster_entity_id] = {
        start_pos = { x = spawn_pos.x, z = spawn_pos.z }, -- grid coordinate at birth
        speed = monster_speed,
        boosted = is_boosted,  -- 继承全局加速状态
        path = nil,
        path_index = 1, -- 当前路径索引
    }

    entity.set_property(monster_entity_id, "robot_profile", monster_profile.class_name or "standard")


    log(string.format("[Spawn] Monster %d spawned with speed: %d", monster_id, monster_speed))

    return monster_entity_id
end


--- 检查候选位置是否与所有已生成宝箱满足最小间距
--- @param cx number 候选格子 x
--- @param cz number 候选格子 z
--- @param min_dist number 最小欧几里得距离
--- @return boolean 满足间距返回 true
local function check_treasure_min_distance(cx, cz, min_dist)
    if min_dist <= 0 then
        return true
    end
    for _, treasure in ipairs(game_state.treasures) do
        local tx = math.floor(treasure.x)
        local tz = math.floor(treasure.z)
        local dx = cx - tx
        local dz = cz - tz
        if math.sqrt(dx * dx + dz * dz) <= min_dist then
            return false
        end
    end
    return true
end

--- 生成宝箱
local function spawn_treasures()
    local treasure_count = usr_config.treasure_count
    -- 软约束：如果存在可用空位，宝箱间距尽量 >15 格；
    -- 若约束过严导致无法生成，则自动降级为仅检查可通行。
    local treasure_min_dist = tonumber(usr_config.treasure_min_distance) or 15.0
    game_state.total_treasures = treasure_count
    
    log(string.format("[Spawn] Spawning %d treasures (min_distance=%s)", treasure_count, tostring(treasure_min_dist)))
    
    for i = 1, treasure_count do
        local attempts = 0
        local max_attempts = 200  -- 增加尝试次数
        local spawned = false
        
        -- 第一轮：is_walkable + 宝箱间距约束
        while attempts < max_attempts and not spawned do
            -- 在更大的范围内随机生成
            local x = random_int(5, game_state.map_width - 5)
            local z = random_int(5, game_state.map_height - 5)
            
            if is_walkable(x, z) and check_treasure_min_distance(x, z, treasure_min_dist) then
                -- 宝箱不阻挡移动，怪物可以穿过
                local treasure_id = entity.spawn("chest_" .. i, x + 0.5, z + 0.5, {"treasure", "collectible"}, {
                    collision = {
                        enabled = false,
                        layer = 1,
                        mask = 0,
                        shape = { type = "circle", radius = 0.3 },
                    },
                })
                if treasure_id then
                    table.insert(game_state.treasures, {
                        id = treasure_id,
                        x = x + 0.5,
                        z = z + 0.5,
                    })
                    spawned = true
                    log(string.format("[Spawn] Treasure %d (id=%d) at grid (%d, %d)", i, treasure_id, x, z))
                end
            end
            
            attempts = attempts + 1
        end

        -- 第二轮降级：仅 is_walkable，忽略间距约束
        if not spawned and treasure_min_dist > 0 then
            log_warn(string.format("[Spawn] Treasure %d: min_distance constraint failed after %d attempts, falling back to no constraint", i, max_attempts))
            local fallback_attempts = 0
            local fallback_max = 50
            while fallback_attempts < fallback_max and not spawned do
                local x = random_int(5, game_state.map_width - 5)
                local z = random_int(5, game_state.map_height - 5)

                if is_walkable(x, z) then
                    local treasure_id = entity.spawn("chest_" .. i, x + 0.5, z + 0.5, {"treasure", "collectible"}, {
                        collision = {
                            enabled = false,
                            layer = 1,
                            mask = 0,
                            shape = { type = "circle", radius = 0.3 },
                        },
                    })
                    if treasure_id then
                        table.insert(game_state.treasures, {
                            id = treasure_id,
                            x = x + 0.5,
                            z = z + 0.5,
                        })
                        spawned = true
                        log(string.format("[Spawn] Treasure %d (id=%d) at grid (%d, %d) [fallback]", i, treasure_id, x, z))
                    end
                end

                fallback_attempts = fallback_attempts + 1
            end
        end
        
        if not spawned then
            log_error(string.format("[Spawn] Failed to spawn treasure %d after all attempts", i))
        end
    end
    
    log(string.format("[Spawn] Successfully spawned %d/%d treasures", #game_state.treasures, treasure_count))
end

--- 检查候选位置是否与所有已生成 buff 满足最小间距
--- @param cx number 候选格子 x
--- @param cz number 候选格子 z
--- @param min_dist number 最小欧几里得距离
--- @return boolean 满足间距返回 true
local function check_buff_min_distance(cx, cz, min_dist)
    if min_dist <= 0 then
        return true
    end
    for _, buff in ipairs(game_state.buffs) do
        local bx = math.floor(buff.x)
        local bz = math.floor(buff.z)
        local dx = cx - bx
        local dz = cz - bz
        if math.sqrt(dx * dx + dz * dz) < min_dist then
            return false
        end
    end
    return true
end

--- 生成加速buff
local function spawn_buffs()
    local buff_count = usr_config.buff_count
    local buff_min_dist = usr_config.buff_min_distance or 0
    
    log(string.format("[Spawn] Spawning %d buffs (min_distance=%s)", buff_count, tostring(buff_min_dist)))
    
    for i = 1, buff_count do
        local attempts = 0
        local max_attempts = 200
        local spawned = false
        
        -- 第一轮：is_walkable + 间距约束
        while attempts < max_attempts and not spawned do
            local x = random_int(5, game_state.map_width - 5)
            local z = random_int(5, game_state.map_height - 5)
            
            if is_walkable(x, z) and check_buff_min_distance(x, z, buff_min_dist) then
                local buff_id = entity.spawn("buff_" .. i, x + 0.5, z + 0.5, {"buff", "speed_buff"}, {
                    collision = {
                        enabled = false,
                        layer = 1,
                        mask = 0,
                        shape = { type = "circle", radius = 0.3 },
                    },
                })
                if buff_id then
                    table.insert(game_state.buffs, {
                        id = buff_id,
                        x = x + 0.5,
                        z = z + 0.5,
                        visible = true,
                    })
                    game_state.buff_refresh_counters[buff_id] = 0
                    spawned = true
                    log(string.format("[Spawn] Buff %d (id=%d) at grid (%d, %d)", i, buff_id, x, z))
                end
            end
            
            attempts = attempts + 1
        end
        
        -- 第二轮降级：仅 is_walkable，忽略间距约束
        if not spawned and buff_min_dist > 0 then
            log_warn(string.format("[Spawn] Buff %d: min_distance constraint failed after %d attempts, falling back to no constraint", i, max_attempts))
            local fallback_attempts = 0
            local fallback_max = 50
            while fallback_attempts < fallback_max and not spawned do
                local x = random_int(5, game_state.map_width - 5)
                local z = random_int(5, game_state.map_height - 5)
                
                if is_walkable(x, z) then
                    local buff_id = entity.spawn("buff_" .. i, x + 0.5, z + 0.5, {"buff", "speed_buff"}, {
                        collision = {
                            enabled = false,
                            layer = 1,
                            mask = 0,
                            shape = { type = "circle", radius = 0.3 },
                        },
                    })
                    if buff_id then
                        table.insert(game_state.buffs, {
                            id = buff_id,
                            x = x + 0.5,
                            z = z + 0.5,
                            visible = true,
                        })
                        game_state.buff_refresh_counters[buff_id] = 0
                        spawned = true
                        log(string.format("[Spawn] Buff %d (id=%d) at grid (%d, %d) [fallback]", i, buff_id, x, z))
                    end
                end
                
                fallback_attempts = fallback_attempts + 1
            end
        end
        
        if not spawned then
            log_error(string.format("[Spawn] Failed to spawn buff %d after all attempts", i))
        end
    end
    
    log(string.format("[Spawn] Successfully spawned %d/%d buffs", #game_state.buffs, buff_count))
end

-- ============================================================================
-- 游戏逻辑
-- ============================================================================

--- 移动智能体（同步执行，立即完成）
local function move_player(direction)
    if not game_state.player then 
        return false 
    end
    
    local dir_vec = DIRECTION_VECTORS[direction]
    if not dir_vec then
        log_error(string.format("[Move] Invalid direction: %d", direction))
        return false
    end
    
    local player = game_state.player
    
    -- 当前玩家所在格子坐标
    local current_grid_x = math.floor(player.x)
    local current_grid_z = math.floor(player.z)
    
    -- 计算移动距离（普通1格，有buff时额外增加 buff_extra_speed 格）
    local move_distance = usr_config.player_speed_normal + (game_state.player_move_bonus or 0)
    if game_state.player_has_speed_buff then
        move_distance = move_distance + usr_config.buff_extra_speed
    end
    
    log(string.format("[Move] Move distance: %d (has_buff: %s)", move_distance, tostring(game_state.player_has_speed_buff)))
    
    -- 逐格移动，找到最后一个可通行的格子，并记录完整路径
    local final_grid_x = current_grid_x
    local final_grid_z = current_grid_z
    local move_path = {}  -- 记录实际走过的路径
    local last_valid_x = current_grid_x
    local last_valid_z = current_grid_z
    
    for step = 1, move_distance do
        local next_grid_x = current_grid_x + dir_vec.x * step
        local next_grid_z = current_grid_z + dir_vec.z * step
        
        -- 使用新的斜向移动检测函数
        if can_move_diagonally(last_valid_x, last_valid_z, next_grid_x, next_grid_z) then
            final_grid_x = next_grid_x
            final_grid_z = next_grid_z
            last_valid_x = next_grid_x
            last_valid_z = next_grid_z
            table.insert(move_path, {x = next_grid_x, z = next_grid_z})
        else
            -- 遇到障碍物，停在前一格
            log(string.format("[Move] Blocked at (%d, %d)", next_grid_x, next_grid_z))
            break
        end
    end
    
    -- 如果没有移动，返回失败
    if final_grid_x == current_grid_x and final_grid_z == current_grid_z then
        return false
    end
    
    -- **立即完成移动（同步执行）**
    entity.move_to(game_state.player.id, final_grid_x + 0.5, final_grid_z + 0.5)
    
    -- 保存路径供收集物品使用
    game_state.player_move_path = move_path
    
    log(string.format("[Move] Moved from (%d, %d) to (%d, %d), path length: %d",
        current_grid_x, current_grid_z, final_grid_x, final_grid_z, #move_path))
    
    return true
end

--- 闪现移动（栅格移动）- 调用技能脚本
local function flash_player(direction)
    -- 调用技能脚本中的闪现逻辑
    local success, flash_path = use_flash_skill(direction, game_state, DIRECTION_VECTORS, is_walkable)
    return success
end

--- 当前动作可用性（legal_act）
--- 返回 16 维数组：
--- - 前 8 维（0-7 移动）固定为 true
--- - 后 8 维（8-15 闪现）仅反映冷却/启用状态，不判断方向可达性
local function can_flash()
    if not game_state.player then
        return false
    end

    -- 冷却中不可闪现
    if game_state.player_flash_cooldown and game_state.player_flash_cooldown > 0 then
        return false
    end

    -- 技能系统可能被禁用
    if skill and skill.get then
        local flash_skill = skill.get("flash")
        if flash_skill and flash_skill.enabled == false then
            return false
        end
    end

    return true
end

local function build_legal_act()
    local mask = {}

    -- 0-7: 移动动作固定可选
    for i = 1, 8 do
        mask[i] = true
    end

    -- 8-15: 闪现动作仅反映冷却/启用状态
    local flash_available = can_flash()
    for dir = 1, 8 do
        mask[8 + dir] = flash_available
    end

    return mask
end

--- 更新怪物AI（每回合玩家行动后，怪物使用A*寻路追逐）
local function update_monsters()
    local function grid_key(x, z)
        return string.format("%d,%d", x, z)
    end

    local function axis_of_move(from_x, from_z, to_x, to_z)
        if from_x ~= to_x then
            return "x"
        end
        if from_z ~= to_z then
            return "z"
        end
        return "none"
    end

    -- 从“所有最短路下一步”中随机选一步，避免 A* 因固定邻居顺序总是先 z 再 x
    local function choose_shortest_path_next_step(cur_x, cur_z, target_x, target_z, occupied, monster_id, state)
        local move_dirs_4 = {
            { 1,  0 },
            { -1, 0 },
            { 0,  1 },
            { 0,  -1 },
        }

        local best_score = nil
        local best = {}

        for _, dv in ipairs(move_dirs_4) do
            local nx = cur_x + dv[1]
            local nz = cur_z + dv[2]
            local nkey = grid_key(nx, nz)
            local occupied_by = occupied[nkey]

            if is_walkable(nx, nz) and (occupied_by == nil or occupied_by == monster_id) then
                local remain_cost = pathfinding.get_distance(nx, nz, target_x, target_z)
                if remain_cost ~= nil then
                    local score = remain_cost

                    -- 防止来回横跳
                    if state.prev_grid_x == nx and state.prev_grid_z == nz then
                        score = score + 0.25
                    end

                    -- 当 x/y 两个方向同样好时，轻微惩罚与上一步同轴，减少“上下对持”
                    local axis = axis_of_move(cur_x, cur_z, nx, nz)
                    if state.last_move_axis == axis then
                        score = score + 0.05
                    end

                    if best_score == nil or score < best_score then
                        best_score = score
                        best = { { x = nx, z = nz, key = nkey, axis = axis } }
                    elseif math.abs(score - best_score) < 1e-6 then
                        table.insert(best, { x = nx, z = nz, key = nkey, axis = axis })
                    end
                end
            end
        end

        if #best == 0 then
            return nil
        end
        return random_choice(best)
    end

    -- 追击候选方向：优先使用8方向，便于绕开局部阻挡，减少“对峙”
    local chase_dirs = {
        { 1,  0 },
        { -1, 0 },
        { 0,  1 },
        { 0,  -1 },
        { 1,  1 },
        { 1,  -1 },
        { -1, 1 },
        { -1, -1 },
    }

    -- 刷新怪物列表
    local all_enemies = entity.get_all_enemies()
    local occupied = {}

    -- 记录本回合开始时怪物占用格（用于避免怪物重叠）
    for _, m in ipairs(all_enemies) do
        local gx = math.floor(m.x)
        local gz = math.floor(m.z)
        local key = grid_key(gx, gz)
        if occupied[key] and occupied[key] ~= m.id then
            log_warn(string.format(
                "[Monster] overlap detected at start: (%d, %d), ids=%d/%d",
                gx,
                gz,
                occupied[key],
                m.id
            ))
        end
        occupied[key] = m.id
    end
    
    if not game_state.player then
        return
    end
    
    for _, monster in ipairs(all_enemies) do
        if not enemy_states[monster.id] then
            goto continue
        end
        
        local state = enemy_states[monster.id]
        local player = game_state.player
        
        -- 当前怪物所在格子
        local current_grid_x = math.floor(monster.x)
        local current_grid_z = math.floor(monster.z)
        local current_key = grid_key(current_grid_x, current_grid_z)
        occupied[current_key] = monster.id
        
        -- 玩家所在格子
        local player_grid_x = math.floor(player.x)
        local player_grid_z = math.floor(player.z)
        
        -- 如果已经在玩家格子上，不再移动
        if current_grid_x == player_grid_x and current_grid_z == player_grid_z then
            goto continue
        end
        
        -- 使用 A* 寻路算法获取路径
        -- 每回合重新计算路径（因为玩家位置会变化）
        local path = pathfinding.find_path(current_grid_x, current_grid_z, player_grid_x, player_grid_z)
        
        if not path or #path < 2 then
            -- 寻路失败：使用“贪心追击 + 防回退惩罚”备选，降低怪物与英雄对峙概率
            log(string.format("[Monster] A* pathfinding failed for monster %d, using greedy chase fallback", monster.id))

            local best_score = nil
            local candidates = {}

            for _, dv in ipairs(chase_dirs) do
                local next_x = current_grid_x + dv[1]
                local next_z = current_grid_z + dv[2]
                local next_key = grid_key(next_x, next_z)
                local occupied_by = occupied[next_key]

                if is_walkable(next_x, next_z) and (occupied_by == nil or occupied_by == monster.id) then
                    -- 对角移动时避免穿角
                    local is_diag = (dv[1] ~= 0 and dv[2] ~= 0)
                    if (not is_diag) or can_move_diagonally(current_grid_x, current_grid_z, next_x, next_z) then
                        local dist = math.max(math.abs(player_grid_x - next_x), math.abs(player_grid_z - next_z))

                        -- 避免在两格之间反复横跳（对上一步位置加轻微惩罚）
                        local backtrack_penalty = 0.0
                        if state.prev_grid_x == next_x and state.prev_grid_z == next_z then
                            backtrack_penalty = 0.25
                        end

                        local score = dist + backtrack_penalty
                        if (best_score == nil) or (score < best_score) then
                            best_score = score
                            candidates = { { x = next_x, z = next_z, key = next_key } }
                        elseif math.abs(score - best_score) < 1e-6 then
                            table.insert(candidates, { x = next_x, z = next_z, key = next_key })
                        end
                    end
                end
            end

            if #candidates > 0 then
                local choice = random_choice(candidates)
                if choice then
                    state.prev_grid_x = current_grid_x
                    state.prev_grid_z = current_grid_z
                    entity.move_to(monster.id, choice.x + 0.5, choice.z + 0.5)
                    occupied[current_key] = nil
                    occupied[choice.key] = monster.id
                    current_key = choice.key
                    current_grid_x = choice.x
                    current_grid_z = choice.z
                    log(string.format("[Monster] Monster %d fallback moved to (%d, %d)", monster.id, choice.x, choice.z))
                end
            end

            goto continue
        end
        
        -- 保存路径到状态（用于调试）
        state.path = path
        state.path_index = 2
        
        -- 沿路径移动 speed 格
        local steps_moved = 0
        for step = 1, state.speed do
            local next_choice = choose_shortest_path_next_step(
                current_grid_x,
                current_grid_z,
                player_grid_x,
                player_grid_z,
                occupied,
                monster.id,
                state
            )

            local next_grid_x = nil
            local next_grid_z = nil
            local next_key = nil
            local next_axis = nil

            if next_choice then
                next_grid_x = next_choice.x
                next_grid_z = next_choice.z
                next_key = next_choice.key
                next_axis = next_choice.axis
            elseif state.path_index <= #state.path then
                -- 兜底：仍可回退到 A* 给出的第一条路径
                local next_point = state.path[state.path_index]
                next_grid_x = next_point.x
                next_grid_z = next_point.z
                next_key = grid_key(next_grid_x, next_grid_z)
                next_axis = axis_of_move(current_grid_x, current_grid_z, next_grid_x, next_grid_z)
            else
                break
            end
            
            -- 检查是否与另一只怪物冲突（同一格子判定）
            local conflict = false
            local occupied_by = occupied[next_key]
            if occupied_by and occupied_by ~= monster.id then
                conflict = true
            end
            
            if not conflict then
                -- 移动到格子中心
                state.prev_grid_x = current_grid_x
                state.prev_grid_z = current_grid_z
                state.last_move_axis = next_axis
                entity.move_to(monster.id, next_grid_x + 0.5, next_grid_z + 0.5)
                occupied[current_key] = nil
                occupied[next_key] = monster.id
                current_key = next_key
                current_grid_x = next_grid_x
                current_grid_z = next_grid_z
                state.path_index = state.path_index + 1
                steps_moved = steps_moved + 1
            else
                log(string.format("[Monster] Monster %d blocked by monster %d at (%d, %d)", 
                    monster.id, occupied_by or -1, next_grid_x, next_grid_z))
                break -- 冲突，停止移动
            end
        end
        
        if steps_moved > 0 then
            log(string.format("[Monster] Monster %d A* moved %d steps to (%d, %d)", 
                monster.id, steps_moved, current_grid_x, current_grid_z))
        end
        
        ::continue::
    end
end

--- 收集宝箱（支持路径检测）
local function collect_treasures()
    if not game_state.player then 
        log_warn("[Collect] Player is nil, cannot collect treasures")
        return 
    end
    
    local player = game_state.player
    local collected = {}
    
    -- 构建检测格子列表
    local check_grids = {}
    
    -- 如果玩家刚完成移动，使用实际走过的路径
    if game_state.player_move_path and #game_state.player_move_path > 0 then
        log(string.format("[Collect] Using move path with %d grids", #game_state.player_move_path))
        for _, grid in ipairs(game_state.player_move_path) do
            table.insert(check_grids, {x = grid.x, z = grid.z})
        end
    else
        -- 没有移动信息，只检查当前位置
        local player_grid_x = math.floor(player.x)
        local player_grid_z = math.floor(player.z)
        table.insert(check_grids, {x = player_grid_x, z = player_grid_z})
    end
    
    log(string.format("[Collect] Checking %d grids for %d treasures", #check_grids, #game_state.treasures))
    
    -- 检查所有路径格子
    for _, grid in ipairs(check_grids) do
        for i, treasure in ipairs(game_state.treasures) do
            -- 获取宝箱所在的格子坐标
            local treasure_grid_x = math.floor(treasure.x)
            local treasure_grid_z = math.floor(treasure.z)
            
            -- 格子判定：玩家必须经过宝箱的格子
            if grid.x == treasure_grid_x and grid.z == treasure_grid_z then
                -- 避免重复收集
                if not collected[i] then
                    log(string.format("[Collect] ✓ Treasure collected at grid (%d, %d)", 
                        treasure_grid_x, treasure_grid_z))
                    
                    -- 增加积分
                    game_state.treasures_collected = game_state.treasures_collected + 1
                    game_state.treasure_score = game_state.treasures_collected * usr_config.treasure_value
                    
                    log(string.format("[Score] Treasures: %d/%d, Treasure Score: %.1f, Total Score: %.1f", 
                        game_state.treasures_collected, game_state.total_treasures,
                        game_state.treasure_score, game_state.score))
                    
                    -- 删除宝箱实体
                    entity.kill(treasure.id)
                    collected[i] = true
                end
            end
        end
    end
    
    -- 从列表中移除已收集的宝箱
    local collected_indices = {}
    for i, _ in pairs(collected) do
        table.insert(collected_indices, i)
    end
    table.sort(collected_indices, function(a, b) return a > b end)
    for _, i in ipairs(collected_indices) do
        table.remove(game_state.treasures, i)
    end
    
    local collected_count = 0
    for _ in pairs(collected) do
        collected_count = collected_count + 1
    end
    
    if collected_count > 0 then
        log(string.format("[Collect] Total collected this step: %d, remaining: %d/%d", 
            collected_count, #game_state.treasures, game_state.total_treasures))
    end
end

--- 收集加速buff（支持路径检测）
local function collect_buffs()
    if not game_state.player then return end
    
    local player = game_state.player
    
    -- 构建检测格子列表
    local check_grids = {}
    
    -- 如果玩家刚完成移动，使用实际走过的路径
    if game_state.player_move_path and #game_state.player_move_path > 0 then
        log(string.format("[Collect] Using move path with %d grids for buffs", #game_state.player_move_path))
        for _, grid in ipairs(game_state.player_move_path) do
            table.insert(check_grids, {x = grid.x, z = grid.z})
        end
    else
        -- 没有移动信息，只检查当前位置
        local player_grid_x = math.floor(player.x)
        local player_grid_z = math.floor(player.z)
        table.insert(check_grids, {x = player_grid_x, z = player_grid_z})
    end
    
    -- 检查所有路径格子
    for _, grid in ipairs(check_grids) do
        for _, buff in ipairs(game_state.buffs) do
            -- 只检查可见的buff
            if buff.visible then
                -- 获取buff所在的格子坐标
                local buff_grid_x = math.floor(buff.x)
                local buff_grid_z = math.floor(buff.z)
                
                -- 格子判定：玩家必须经过buff的格子
                if grid.x == buff_grid_x and grid.z == buff_grid_z then
                    log(string.format("[Collect] ✓ Buff collected at grid (%d, %d)", buff_grid_x, buff_grid_z))
                    
                    -- 激活加速buff
                    game_state.player_has_speed_buff = true
                    game_state.player_speed_buff_remaining = usr_config.buff_duration
                    game_state.buff_obtained_count = (game_state.buff_obtained_count or 0) + 1
                    
                    -- 隐藏buff（视觉上消失）

                    buff.visible = false
                    entity.set_visible(buff.id, false)
                    
                    -- 触发刷新冷却
                    game_state.buff_refresh_counters[buff.id] = usr_config.buff_refresh_time
                    
                    log(string.format("[Buff] Speed buff activated, duration: %d steps, will refresh after: %d steps", 
                        usr_config.buff_duration, usr_config.buff_refresh_time))
                    
                    -- 只激活第一个buff就退出
                    return
                end
            end
        end
    end
end

--- 更新buff冷却
local function update_buffs()
    -- 更新加速buff持续时间
    if game_state.player_has_speed_buff then
        game_state.player_speed_buff_remaining = game_state.player_speed_buff_remaining - 1
        if game_state.player_speed_buff_remaining <= 0 then
            game_state.player_has_speed_buff = false
            log("[Buff] Speed buff expired")
        end
    end
    
    -- 更新buff刷新计数器
    for _, buff in ipairs(game_state.buffs) do
        local counter = game_state.buff_refresh_counters[buff.id]
        if counter and counter > 0 then
            game_state.buff_refresh_counters[buff.id] = counter - 1
            if game_state.buff_refresh_counters[buff.id] == 0 then
                -- buff刷新完成，重新显示
                buff.visible = true
                entity.set_visible(buff.id, true)
                log(string.format("[Buff] Buff id=%d refreshed and available at grid (%d, %d)", 
                    buff.id, math.floor(buff.x), math.floor(buff.z)))
            end
        end
    end
    
    -- 更新闪现冷却
    if game_state.player_flash_cooldown > 0 then
        game_state.player_flash_cooldown = game_state.player_flash_cooldown - 1
    end
    
    -- 同步技能系统的冷却时间
    skill.set_cooldown("flash", game_state.player_flash_cooldown)
end

--- 生成第二个怪物
--- 规则：
--- - 出现时机：默认 300 步（usr_config.monster_interval 可配置）
--- - 初始位置：玩家 10 步前所在格子（若该格不可走则向周围找可走点）
local function spawn_second_monster()
    if game_state.second_enemy_spawned then
        return
    end

    -- 如果配置了不生成第二个怪物，直接返回
    if usr_config.monster_interval == -2 or usr_config.monster_count < 2 then
        log(string.format(
            "[Spawn] Second monster disabled (monster_interval=%s, monster_count=%s)",
            tostring(usr_config.monster_interval),
            tostring(usr_config.monster_count)
        ))
        game_state.second_enemy_spawned = true -- 标记为已完成，避免重复检查
        return
    end


    if game_state.current_step >= game_state.second_enemy_spawn_step then
        local pos10 = get_player_pos_steps_ago(10)
        if not pos10 and game_state.player then
            pos10 = {
                 x = math.floor(game_state.player.x),
                 z = math.floor(game_state.player.z),
            }
        end

        if not pos10 then
            log_warn("[Spawn] Failed to locate player's position history for monster2")
            game_state.second_enemy_spawned = true
            return
        end

        log(string.format("[Spawn] Spawning second monster at player's 10-steps-ago pos (%d,%d)", pos10.x, pos10.z))
        local monster_eid = spawn_monster(2, pos10)
        if monster_eid then
            table.insert(game_state.enemies, monster_eid)
            game_state.second_enemy_spawned = true
        else
            log_warn("[Spawn] Failed to spawn second monster")
            game_state.second_enemy_spawned = true -- 标记为完成，避免重复尝试
        end
    end
end


--- 怪物加速增益
local function boost_monsters()
    if game_state.monster_speed_boosted then return end
    
    -- -- 如果配置了-1，随机范围1～2000
    -- if usr_config.monster_speed_boost_step == -1 then
    --     usr_config.monster_speed_boost_step = random_int(1, 2000)
    -- end
    
    if game_state.current_step >= game_state.monster_speed_boost_step then
        log("[Monster] Speed boost activated")
        
        for _, monster_id in ipairs(game_state.enemies) do
            if enemy_states[monster_id] then
                enemy_states[monster_id].speed = enemy_states[monster_id].speed + 1
                enemy_states[monster_id].boosted = true
            end
        end
        
        game_state.monster_speed_boosted = true
    end
end

--- 计算得分
local function calculate_score()
    game_state.step_score = game_state.current_step * usr_config.step_score_multiplier
    game_state.score = game_state.step_score + game_state.treasure_score
end

-- ============================================================================
-- 观测生成（21x21视野域）
-- ============================================================================

--- 获取智能体观测
function get_agent_observation()
    if not game_state.player then
        log_error("[Observation] Player not found")
        return nil
    end
    
    -- 刷新player引用
    game_state.player = entity.get_player()
    if not game_state.player then
        log_error("[Observation] Failed to refresh player")
        return nil
    end
    
    local player = game_state.player
    local player_z = player.z
    local vision_range = usr_config.vision_range
    
    local obs = {
        -- 智能体自身状态
        player = {
            x = player.x,
            z = player_z,
            has_speed_buff = game_state.player_has_speed_buff,
            speed_buff_remaining = game_state.player_speed_buff_remaining,
            flash_cooldown = game_state.player_flash_cooldown,
        },
        
        -- 怪物信息（相对位置）
        enemies = {},
        
        -- 宝箱信息（相对位置）
        treasures = {},
        
        -- buff信息（相对位置）
        buffs = {},
        
        -- 地图信息（21x21视野域）
        vision_grid = {},
        
        -- 任务状态
        task = {
            current_step = game_state.current_step,
            max_step = game_state.max_step,
            score = game_state.score,
            treasures_collected = game_state.treasures_collected,
            total_treasures = game_state.total_treasures,
        },
    }
    
    -- 收集怪物信息
    local all_enemies = entity.get_all_enemies()
    for _, monster in ipairs(all_enemies) do
        local monster_z = monster.z
        local rel_x = monster.x - player.x
        local rel_z = monster_z - player_z
        
        -- 只包含视野内的怪物
        if math.abs(rel_x) <= vision_range and math.abs(rel_z) <= vision_range then
            local state = enemy_states[monster.id]
            table.insert(obs.enemies, {
                rel_x = rel_x,
                rel_z = rel_z,
                speed = state and state.speed or 1,
                boosted = state and state.boosted or false,
            })
        end
    end
    
    -- 收集宝箱信息
    for _, treasure in ipairs(game_state.treasures) do
        local rel_x = treasure.x - player.x
        local rel_z = treasure.z - player_z
        
        if math.abs(rel_x) <= vision_range and math.abs(rel_z) <= vision_range then
            table.insert(obs.treasures, {
                rel_x = rel_x,
                rel_z = rel_z,
            })
        end
    end
    
    -- 收集buff信息
    for _, buff in ipairs(game_state.buffs) do
        local rel_x = buff.x - player.x
        local rel_z = buff.z - player_z
        
        if math.abs(rel_x) <= vision_range and math.abs(rel_z) <= vision_range then
            table.insert(obs.buffs, {
                rel_x = rel_x,
                rel_z = rel_z,
                available = buff.visible,  -- 使用visible状态
            })
        end
    end
    
    -- 生成视野栅格（21x21）
    for dy = -vision_range, vision_range do
        local row = {}
        for dx = -vision_range, vision_range do
            local x = math.floor(player.x) + dx
            local z = math.floor(player_z) + dy
            
            if is_walkable(x, z) then
                table.insert(row, 1) -- 可通行
            else
                table.insert(row, 0) -- 障碍物
            end
        end
        table.insert(obs.vision_grid, row)
    end
    
    return obs
end

-- ============================================================================
-- 动作执行
-- ============================================================================

--- 执行智能体动作（同步执行，立即完成）
function execute_agent_action(action)
    if game_state.task_status ~= "running" then
        log_warn("[Action] Task not running, ignoring action")
        return false
    end
    
    local success = false
    
    -- 判断是移动还是闪现
    if action >= 0 and action <= 7 then
        -- 移动动作（同步执行）
        success = move_player(action)
    elseif action >= 8 and action <= 15 then
        -- 闪现动作（同步执行）
        local direction = action - 8
        success = flash_player(direction)
        if success then
            game_state.skill_usage_count = (game_state.skill_usage_count or 0) + 1
        end
    else
        log_error(string.format("[Action] Invalid action: %d", action))
        return false
    end
    
    -- 如果执行成功，立即触发后续游戏逻辑

    -- 刷新玩家引用
    game_state.player = entity.get_player()
    if not game_state.player then
        log_error("[Action] Failed to refresh player")
        return false
    end
    
    -- 步数增加
    game_state.current_step = game_state.current_step + 1

    -- 记录当前位置（用于：怪物2按玩家10步前位置生成）
    push_player_pos_history()
    
    -- 收集物品

    collect_treasures()
    collect_buffs()
    
    -- 清除移动路径（避免下次误用）
    game_state.player_move_path = nil
    
    -- 怪物移动
    update_monsters()
    
    -- 生成第二个怪物、怪物加速
    spawn_second_monster()
    boost_monsters()
    
    -- 更新 buff 冷却
    update_buffs()
    
    -- 碰撞检测
    if GorgeEnemyElement.fail_if_enemy_near_player(game_state, {
        range = 1,
        require_running = false,
    }) then
        log("[Task] Task failed: player caught by monster")
    end
    
    -- 检查是否完成任务
    if game_state.current_step >= game_state.max_step then
        log("current_stape"..game_state.current_step)
        log( "game_state.max_step"..game_state.max_step)
        game_state.task_status = "completed"
        log("[Task] Task completed: player survived max_step rounds!")
    end
    
    -- 计算得分
    calculate_score()
    
end

-- ============================================================================
-- 任务状态查询
-- ============================================================================

--- 获取任务状态
local function build_status_api()
    return GorgeChaseStatus.new({
        get_game_state = function() return game_state end,
        get_usr_config = function() return usr_config end,
        calculate_score = calculate_score,
    })
end

function get_task_state()
    return build_status_api().get_task_state()
end

function get_hud_text()
    return build_status_api().get_hud_text()
end


-- ============================================================================
-- FrameState协议构建函数（符合开悟协议，需在reset之前定义）
-- ============================================================================

local _frame_builder = nil

local function ensure_frame_builder()
    if _frame_builder then
        return _frame_builder
    end

    _frame_builder = GorgeChaseFrame.new({
        entity = entity,
        log = log,
        get_game_state = function() return game_state end,
        get_usr_config = function() return usr_config end,
        get_enemy_states = function() return enemy_states end,
        get_system_config = function() return usr_config end,
        is_walkable = is_walkable,
        calc_hero_l2_distance_bucket = calc_hero_l2_distance_bucket,
        calc_hero_relative_direction_id = calc_hero_relative_direction_id,
    })

    return _frame_builder
end

local function build_protocol_obs(frame_no, terminated, truncated, result_message)
    local frame_builder = ensure_frame_builder()
    local frame_state = frame_builder.build_frame_state()
    local env_info = frame_builder.build_env_info()
    local map_id = game_state.current_map_id or 0
    local env_id = tostring(map_id)

    return {
        env_id = env_id,
        frame_no = frame_no,
        observation = {
            step_no = frame_no,
            frame_state = frame_state,
            env_info = env_info,
            map_info = frame_builder.build_map_info(),
            legal_act = build_legal_act(),
        },
        extra_info = {
            frame_state = frame_state,
            map_id = map_id,
            result_code = 0,
            result_message = result_message or "",
        },
        terminated = terminated,
        truncated = truncated,
    }
end

-- ============================================================================
-- 生命周期函数
-- ============================================================================


--- 初始化任务（由Python调用，传入usr_config）
function reset(user_config)
    log("========================================")
    log("[Reset] Gorge Chase - Task Reset")
    log("========================================")

    -- 合并用户配置与默认配置（用户配置覆盖默认值）
    usr_config = merge_config(DEFAULT_CONFIG, user_config)
    
    -- 初始化随机种子（必须在所有随机操作之前）
    -- 约定：random_seed>0 时使用外部传入值；random_seed==-1 时自动生成随机种子；其他情况保持当前随机状态不变
    local seed = tonumber(usr_config.random_seed)
    if seed ~= nil and seed > 0 then
        seed = math.floor(seed + 1e-6)
        math.randomseed(seed)
    elseif seed == -1 then
        local time_seed = os.time()
        local clock_seed = math.floor((os.clock() % 1) * 1000000)
        seed = time_seed + clock_seed
        math.randomseed(seed)
    else
        seed = nil
    end
    log(string.format("[Reset] Random seed initialized: %s", tostring(seed)))
    
    -- 验证并修正配置
    if not validate_config(usr_config) then
        log_error("[Reset] Invalid configuration")
        game_state.task_status = "failed"
        return {
            obs = {
                env_id = "0",
                frame_no = 0,
                observation = {},
                extra_info = {
                    frame_state = {},
                    map_id = (usr_config and usr_config.map_id) or 0,
                    result_code = -1,
                    result_message = "Invalid configuration",
                },
                terminated = true,
                truncated = false,
            }
        }
    end

    -- 兼容注释约定：monster_interval=-1 表示 0-1000 随机；-2 表示不生成第二只
    if usr_config.monster_interval == -1 then
        usr_config.monster_interval = random_int(11, 2000)
        log(string.format("[Config] Random monster_interval: %d", usr_config.monster_interval))
    end

    -- 如果配置了-1，随机范围1～2000
    if usr_config.monster_speed_boost_step == -1 then
        usr_config.monster_speed_boost_step = random_int(1, 2000)
    end

    log(string.format(
        "[Config] Runtime => player_speed_normal=%s, treasure_value=%s, step_score_multiplier=%s, vision_range=%s",
        tostring(usr_config.player_speed_normal),
        tostring(usr_config.treasure_value),
        tostring(usr_config.step_score_multiplier),
        tostring(usr_config.vision_range)
    ))


    -- 可观测性：打印“脚本归一化后的配置键”
    do
        local keys = collect_sorted_keys(usr_config)
        log(string.format("[Config] Normalized keys(%d): %s", #keys, table.concat(keys, ", ")))
    end
    key_diff_report(__gorge_chase_injected_keys, usr_config, "gorge_chase")
    
    -- 打印配置

    log("[Config] Task configuration:")
    for k, v in pairs(usr_config) do
        log(string.format("  %s = %s", k, tostring(v)))
    end
    
    -- 重置游戏状态
    game_state = {
        player = nil,
        enemies = {},
        treasures = {},
        buffs = {},
        
        current_step = 0,
        max_step = usr_config.max_step,
        task_status = "running",
        
        score = 0,
        step_score = 0,
        treasure_score = 0,
        treasures_collected = 0,
        total_treasures = 0,
        skill_usage_count = 0,
        buff_obtained_count = 0,
        
        player_has_speed_buff = false,

        player_speed_buff_remaining = 0,
        player_flash_cooldown = 0,
        player_move_bonus = 0,
        
        second_enemy_spawned = false,
        second_enemy_spawn_step = usr_config.monster_interval or 300,
        monster_speed_boosted = false,

        monster_speed_boost_step = usr_config.monster_speed_boost_step,
        
        buff_refresh_counters = {},
        
        -- 从地图API读取实际宽高
        map_width = map.get_width() or 128,
        map_height = map.get_height() or 128,
        current_map_id = usr_config.map_id,
        
        -- 移动状态（单帧同步）
        player_move_path = nil,
        monster_speed_bonus = 0,

        -- 玩家位置历史（用于：怪物2出生点）
        player_pos_history = {},

        -- 调试面板：每次 reset 都清空，避免停止后再次播放出现状态残留
        map_viz_visible = false,
        last_map_viz_grid = nil,
        last_map_viz_cx = nil,
        last_map_viz_cz = nil,
        last_map_viz_vr = nil,
    }

    
    log(string.format("[Reset] Map size from API: %dx%d", game_state.map_width, game_state.map_height))
    
    enemy_states = {}
    
    -- 清理所有旧实体
    log("[Reset] Cleaning up old entities...")
    entity.clear()
    
    -- 初始化技能系统
    init_skills()
    
    -- 注意：地图已由 Python 端在调用 reset 之前加载
    -- map.get_width() 和 map.get_height() 会返回当前加载地图的尺寸
    
    -- 生成实体
    log("[Reset] Spawning entities...")
    
    -- 先生成宝箱和buff，确保玩家出生点可以排除这些位置
    spawn_treasures()
    spawn_buffs()
    
    if not spawn_player() then
        log_error("[Reset] Failed to spawn player")
        return {
            obs = {
                env_id = "0",
                frame_no = 0,
                observation = {},
                extra_info = {
                    frame_state = {},
                    map_id = (game_state and game_state.current_map_id) or (usr_config and usr_config.map_id) or 0,
                    result_code = -2,
                    result_message = "Failed to spawn player",
                },
                terminated = true,
                truncated = false,
            }
        }
    end

    -- 初始化位置历史（step=0）
    push_player_pos_history()
    
    -- 根据配置生成初始怪物
    if usr_config.monster_count >= 1 then
        local first_monster = spawn_monster(1)
        if first_monster then
            table.insert(game_state.enemies, first_monster)
            log("[Reset] First monster spawned")
        else
            log_warn("[Reset] Failed to spawn first monster (may be no spawn points)")
        end
    else
        log("[Reset] monster_count = 0, no monsters will spawn")
    end

    -- 部署/初始化信息（对齐其他玩法的输出习惯）
    log(string.format(
        "[Init] map=%dx%d, monsters=%d, treasures=%d, buffs=%d, max_step=%d",
        game_state.map_width,
        game_state.map_height,
        #game_state.enemies,
        #game_state.treasures,
        #game_state.buffs,
        usr_config.max_step or 0
    ))
    log(string.format(
        "[Deploy] map_random=%s, map_id=%s, flash_cd=%s, flash_dist=%s/%s",
        tostring(usr_config.map_random),
        tostring(usr_config.map_id),
        tostring(usr_config.flash_cooldown),
        tostring(usr_config.flash_distance),
        tostring(usr_config.flash_distance_diagonal)
    ))
    if __gorge_chase_user_config_path ~= nil then
        log(string.format("[Deploy] config=%s", tostring(__gorge_chase_user_config_path)))
    end

    log("[Reset] Task reset complete")

    if game_state.player then
        log(string.format("[Reset] Map: %d, Player: (%.1f, %.1f), Monsters: %d, Treasures: %d, Buffs: %d",
            game_state.current_map_id,
            game_state.player.x,
            game_state.player.z,
            #game_state.enemies,
            #game_state.treasures,
            #game_state.buffs))
    else
        log("[Reset] Player spawn failed")
        return {
            obs = {
                env_id = "0",
                frame_no = 0,
                observation = {},
                extra_info = {
                    frame_state = {},
                    map_id = (game_state and game_state.current_map_id) or (usr_config and usr_config.map_id) or 0,
                    result_code = -3,
                    result_message = "Player state invalid",
                },
                terminated = true,
                truncated = false,
            }
        }
    end
    
    return {
        obs = build_protocol_obs(0, false, false, "success")
    }
end

-- ============================================================================
-- 初始化
-- ============================================================================

local _world_api = nil

local function ensure_world_api()
    if _world_api then
        return _world_api
    end

    _world_api = GorgeChaseWorld.new({
        log = log,
        control = Control,
        entity = entity,
        reset = reset,
        execute_agent_action = execute_agent_action,
        collect_frame_state = collect_frame_state,
        get_game_state = function() return game_state end,
        get_step_action = function() return _step_action end,
        clear_step_action = function()
            _step_action = nil
            _step_extra_info = nil
        end,
        frame_builder = {
            visualize_map_info_grid = function(d)
                local p = entity and entity.get_player and entity.get_player()
                if not p then return false end
                local cx = math.floor(p.x)
                local cz = math.floor(p.z)
                local vr = (d and d.vision_range) or 10
                local frame_builder = ensure_frame_builder()
                local grid_data = frame_builder.build_map_info()
                game_state.last_map_viz_grid = grid_data
                game_state.last_map_viz_cx = cx
                game_state.last_map_viz_cz = cz
                game_state.last_map_viz_vr = vr
                return true
            end,
            clear_map_info_visualization = function()
                game_state.last_map_viz_grid = nil
                game_state.last_map_viz_cx = nil
                game_state.last_map_viz_cz = nil
                game_state.last_map_viz_vr = nil
            end,
        },
    })

    return _world_api
end

--- 启动回调
function on_start()
    return ensure_world_api().on_start()
end


--- 更新玩家移动动画
--- 更新回调（每帧调用）
--- 支持强化学习的单步同步执行
function on_update(dt)
    return ensure_world_api().on_update(dt)
end

--- 停止回调
function on_stop()
    return ensure_world_api().on_stop()
end

-- ============================================================================
-- Python API 接口（标准RL接口）
-- ============================================================================

--- collect_frame_state - 返回符合开悟协议的数据
--- @return table {reward, obs}
function collect_frame_state()
    -- 计算奖励
    local reward = 0
    -- if success then
    --     reward = game_state.score or 0
    -- end
    
    -- 判断是否结束
    local truncated = (game_state.current_step >= game_state.max_step)
    local done = (game_state.task_status == "completed" or game_state.task_status == "failed") and (not truncated)

    
    local frame_no = game_state.current_step or 0
    local env_id = tostring(game_state.current_map_id or 0)
    
    return {
        reward = {
            env_id = env_id,
            frame_no = frame_no,
            reward = reward,
        },
        obs = build_protocol_obs(frame_no, done, truncated, ""),
    }
end

--- 获取所有实体状态（供Python调用）
--- @return table 实体状态列表
function get_all_entity_states()
    local states = {}
    
    -- 获取玩家
    local player = entity.get_player()
    if player then
        table.insert(states, {
            id = player.id,
            type = "player",
            name = player.name or "player",
            position = {x = player.x, z = player.z},
            facing = "east",
            alive = true,
        })
    end
    
    -- 获取所有敌人
    local enemies = entity.get_all_enemies()
    for _, e in ipairs(enemies) do
        local state = enemy_states[e.id]
        table.insert(states, {
            id = e.id,
            type = "enemy",
            name = e.name or "enemy",
            position = {x = e.x, z = e.z},
            facing = "east",
            alive = true,
            ai_state = state and {
                state = state.state or "idle",
                chase_range = state.chase_range or 0,
                speed = state.speed or 1,
            } or nil,
        })
    end
    
    return states
end

--- 获取游戏状态（供Python调用）
--- @return table 游戏状态
function get_game_state()
    return {
        game_over = (game_state.task_status == "completed" or game_state.task_status == "failed"),
        task_status = game_state.task_status,
        entity_count = entity.count(),
        current_step = game_state.current_step,
        max_step = game_state.max_step,
        score = game_state.score,
        config = {
            use_pathfinding = true,
            vision_range = usr_config.vision_range,
        },
    }
end

-- Rust 调试面板读取入口（全局函数）
function get_map_viz_grid()
    return {
        grid = game_state.last_map_viz_grid,
        cx = game_state.last_map_viz_cx,
            cz = game_state.last_map_viz_cz,
        vr = game_state.last_map_viz_vr,
    }
end

log("Gorge Chase script loaded successfully")
