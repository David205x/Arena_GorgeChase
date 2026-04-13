-- ============================================================================
-- 闪现技能 (Flash Skill)
-- ============================================================================
-- 该技能允许玩家快速移动到远处：忽略中间阻挡，只要落点格子可通行就允许闪现


-- ============================================================================
-- 技能配置
-- ============================================================================
local FLASH_CONFIG = {
    id = "flash",
    name = "闪现",
    description = "快速移动到远处",
    icon = "⚡",
    cooldown = 100,
    distance = 10,           -- 直线方向闪现距离
    distance_diagonal = 8,   -- 斜向闪现距离
}

-- ============================================================================
-- 技能核心逻辑
-- ============================================================================

--- 执行闪现
--- @param direction number 闪现方向（0-7：右、右上、上、左上、左、左下、下、右下）
--- @param game_state table 游戏状态
--- @param direction_vectors table 方向向量表
--- @param is_walkable_func function 可通行性检查函数
--- @return boolean, table 是否成功, 移动路径
local function execute_flash(direction, game_state, direction_vectors, is_walkable_func)
    if not game_state.player then 
        log_error("[Flash] Player is nil")
        return false, {}
    end
    
    -- 检查冷却
    if game_state.player_flash_cooldown > 0 then
        log_debug(string.format("[Flash] Still in cooldown: %d steps remaining", game_state.player_flash_cooldown))
        return false, {}
    end
    
    local dir_vec = direction_vectors[direction]
    if not dir_vec then
        log_error(string.format("[Flash] Invalid direction: %d", direction))
        return false, {}
    end
    
    local player = game_state.player
    
    -- 当前玩家所在格子坐标
    local current_grid_x = math.floor(player.x)
    local current_grid_z = math.floor(player.z)
    
    -- 判断是否斜向
    local is_diagonal = (dir_vec.x ~= 0 and dir_vec.z ~= 0)
    local flash_distance = is_diagonal and FLASH_CONFIG.distance_diagonal or FLASH_CONFIG.distance
    
    -- 计算范围内“最远可通行落点”（忽略中间阻挡）
    local final_step = nil
    local final_grid_x = current_grid_x
    local final_grid_z = current_grid_z

    for step = flash_distance, 1, -1 do
        local check_grid_x = current_grid_x + dir_vec.x * step
        local check_grid_z = current_grid_z + dir_vec.z * step

        if is_walkable_func(check_grid_x, check_grid_z) then
            final_step = step
            final_grid_x = check_grid_x
            final_grid_z = check_grid_z
            break
        end
    end

    -- 范围内没有任何可通行格子，执行原地闪现（不移动，但消耗冷却）
    if not final_step then
        log("[Flash] No walkable target within range, flash in place (consuming cooldown)")
        return true, {}
    end

    -- 记录闪现路径（仅用于表现/收集逻辑；路径中可能包含不可通行格子）
    local flash_path = {}
    for step = 1, final_step do
        local path_x = current_grid_x + dir_vec.x * step
        local path_z = current_grid_z + dir_vec.z * step
        table.insert(flash_path, {x = path_x, z = path_z})
    end

    -- 闪现到最终位置
    local target_x = final_grid_x + 0.5
    local target_z = final_grid_z + 0.5

    entity.move_to(player.id, target_x, target_z)

    log(string.format("[Flash] Player flashed from grid (%d, %d) to grid (%d, %d), distance: %d", 
        current_grid_x, current_grid_z, final_grid_x, final_grid_z, final_step))

    return true, flash_path

end

-- ============================================================================
-- 对外接口
-- ============================================================================

--- 注册闪现技能
--- @param config table 配置参数 {cooldown, distance, distance_diagonal}
function register_flash_skill(config)
    -- 合并配置
    if config then
        FLASH_CONFIG.cooldown = config.cooldown or FLASH_CONFIG.cooldown
        FLASH_CONFIG.distance = config.distance or FLASH_CONFIG.distance
        FLASH_CONFIG.distance_diagonal = config.distance_diagonal or FLASH_CONFIG.distance_diagonal
        FLASH_CONFIG.description = string.format("快速移动到远处，冷却%d步", FLASH_CONFIG.cooldown)
    end
    
    -- 注册到技能系统
    skill.register(
        FLASH_CONFIG.id,
        FLASH_CONFIG.name,
        FLASH_CONFIG.description,
        FLASH_CONFIG.icon,
        FLASH_CONFIG.cooldown
    )
    
    log(string.format("[Skill] Flash skill registered (cooldown: %d, distance: %d/%d)", 
        FLASH_CONFIG.cooldown, FLASH_CONFIG.distance, FLASH_CONFIG.distance_diagonal))
end

--- 使用闪现技能（从玩法脚本调用）
--- @param direction number 闪现方向
--- @param game_state table 游戏状态
--- @param direction_vectors table 方向向量表
--- @param is_walkable_func function 可通行性检查函数
--- @return boolean, table 是否成功, 移动路径
function use_flash_skill(direction, game_state, direction_vectors, is_walkable_func)
    local success, flash_path = execute_flash(direction, game_state, direction_vectors, is_walkable_func)
    
    if success then
        -- 保存闪现路径（用于收集物品）
        game_state.player_move_path = flash_path
        
        -- 刷新player引用
        game_state.player = entity.get_player()
        
        -- 触发冷却
        game_state.player_flash_cooldown = FLASH_CONFIG.cooldown
        
        -- 触发技能冷却
        skill.use_skill(FLASH_CONFIG.id)
    end
    
    return success, flash_path
end

--- 获取闪现配置
function get_flash_config()
    return FLASH_CONFIG
end

return {
    register_flash_skill = register_flash_skill,
    use_flash_skill = use_flash_skill,
    get_flash_config = get_flash_config,
}

