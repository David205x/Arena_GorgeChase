import numpy as np
import time
import os
from kaiwu_agent.utils.common_func import Frame, attached

from tools.train_env_conf_validate import read_usr_conf
from tools.metrics_utils import get_training_metrics
from agent_ppo.feature.definition import (
    SampleManager,
)
from agent_ppo.conf.conf import Config


@attached
def workflow(envs, agents, logger=None, monitor=None):
    try:
        env, agent = envs[0], agents[0]
        episode_num_every_epoch = 1
        last_save_model_time = 0
        last_put_data_time = 0
        monitor_data = {}

        # 配置文件读取和校验
        usr_conf = read_usr_conf('agent_ppo/conf/train_env_conf.toml', logger)
        if usr_conf is None:
            logger.error(
                f'usr_conf is None, please check agent_ppo/conf/train_env_conf.toml'
            )
            return

        while True:
            for g_data, monitor_data in run_episodes(
                episode_num_every_epoch, env, agent, usr_conf, logger, monitor
            ):
                agent.learn(g_data)
                g_data.clear()

            # 保存model文件
            now = time.time()
            if now - last_save_model_time >= Config.SAVE_INTERVAL:
                agent.save_model()
                last_save_model_time = now

            # 上报监控指标
            if now - last_put_data_time >= Config.LOG_INTERVAL:
                monitor.put_data({os.getpid(): monitor_data})
                last_put_data_time = now

    except Exception as e:
        raise RuntimeError(f'workflow error')


def run_episodes(n_episode, env, agent, usr_conf, logger, monitor):
    try:
        for episode in range(n_episode):
            collector = SampleManager()
            win_rate = 0

            # 获取训练中的指标
            training_metrics = get_training_metrics()
            if training_metrics:
                logger.info(f'training_metrics is {training_metrics}')

            # 重置任务, 并获取初始状态
            obs, extra_info = env.reset(usr_conf=usr_conf)
            if extra_info['result_code'] < 0:
                logger.error(
                    f'env.reset result_code is {extra_info["result_code"]}, result_message is {extra_info["result_message"]}'
                )
                raise RuntimeError(extra_info['result_message'])
            elif extra_info['result_code'] > 0:
                continue

            # 每次对局开始时, 支持加载最新model文件, 该调用会从远程的训练节点加载最新模型
            agent.reset()
            agent.load_model(id='latest')

            done = False
            step = 0
            episodic_return = 0

            max_step_no = int(os.environ.get('max_step_no', '0'))

            while not done:
                # 特征处理
                obs_data = agent.observation_process(obs)

                # Agent 进行推理, 获取下一帧的预测动作
                act_data, model_version = agent.predict(list_obs_data=[obs_data])

                # ActData 解包成动作
                act = agent.action_process(act_data[0])

                # 与环境交互, 执行动作, 获取下一步的状态
                step_no, _obs, terminated, truncated, _extra_info = env.step(act)
                if _extra_info['result_code'] != 0:
                    logger.warning(
                        f'_extra_info.result_code is {_extra_info["result_code"]}, \
                        _extra_info.result_message is {_extra_info["result_message"]}'
                    )
                    break

                step += 1

                reward = agent.reward_process(extra_info, _extra_info)

                # 判断任务结束, 并更新胜利次数
                game_info = _extra_info['game_info']
                final_reward = 0.0
                if truncated:
                    win_rate = agent.update_win_rate(False)
                    final_reward += Config.REWARD_WEIGHTS['reward_lose']
                    logger.info(
                        f'Game truncated! step_no:{step_no} score:{game_info["total_score"]} win_rate:{win_rate}'
                    )
                elif terminated:
                    win_rate = agent.update_win_rate(True)
                    final_reward += Config.REWARD_WEIGHTS['reward_win']
                    logger.info(
                        f'Game terminated! step_no:{step_no} score:{game_info["total_score"]} win_rate:{win_rate}'
                    )
                reward += final_reward
                done = (
                    terminated or truncated or (max_step_no > 0 and step >= max_step_no)
                )

                episodic_return += reward
                # 构造任务帧，为构造样本做准备
                collector.sample_process(
                    feature=obs_data.feature,
                    legal_action=obs_data.legal_action,
                    prob=[act_data[0].prob],
                    action=[act_data[0].action],
                    value=act_data[0].value,
                    reward=np.array([reward]),
                )
                # 如果任务结束，则进行样本处理，将样本送去训练
                if done:
                    monitor_data = {
                        'diy_1': win_rate,
                        'diy_2': episodic_return,
                    }
                    yield collector.finalize_trajectory(), monitor_data
                    break
                elif step % Config.LOAD_MODEL_INTERVAL == 0:
                    agent.load_model(id='latest')

                # 状态更新
                obs = _obs
                extra_info = _extra_info

    except Exception as e:
        logger.error(f'run_episodes error')
        raise RuntimeError(f'run_episodes error')
