# -*- coding: utf-8 -*-
"""规则专家：根据真实任务几何关系直接给出正确动作标签。"""

import numpy as np


INSTRUCTION = "Push the blue cube to the green target."


def scripted_expert_action(env):
    """
    专家知道方块和目标的精确位置，因此可以直接计算推送方向。

    这个函数不学习，也没有神经网络参数。它的作用是生成 demonstration：
        observation_t -> expert action_t
    """
    push_direction = env.get_push_direction()

    # xy沿目标方向推动；z保持为0，让末端主要完成桌面平面运动。
    action = np.array(
        [push_direction[0], push_direction[1], 0.0], dtype=np.float32
    )
    return np.clip(action, -1.0, 1.0)

