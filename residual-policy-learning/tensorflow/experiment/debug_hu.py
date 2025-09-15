# import gym
# import time
# import numpy as np
# env = gym.make("UR5ePickAndPlace-v2")
# obs  = env.reset()
# print("UR5e 初始位置:", obs['observation'][:3])  # 检查 x, y, z 位置
# init_pos = obs['observation'][:3]
# action_init = np.concatenate([init_pos, np.array([0])])
# for i in range(1000):
#     action = env.action_space.sample()
#     # import pdb; pdb.set_trace()
#     # env.step(action)
#     env.step(action_init)
#     env.render()
#     time.sleep(0.1)
# print(env.action_space)

import gym
import time
import numpy as np

env = gym.make("UR5ePickAndPlace-v2")
obs  = env.reset()

# STEP 1: 观察初始位置
print("UR5e 初始位置 (observation[:3]):", obs['observation'][:3])

# STEP 2: 获取 mocap 应绑定的 body 位姿
body_name = "robot0:2f85:base"  # 或 robot0:2f85:pinch 也可以试试
body_pos = env.sim.data.get_body_xpos(body_name)
body_quat = env.sim.data.get_body_xquat(body_name)

print("应设置的 <body mocap='true'> 的位置 pos =", body_pos)
print("应设置的 <body mocap='true'> 的四元数 quat =", body_quat)

# STEP 3: 固定动作测试
action_init = np.concatenate([obs['observation'][:3], np.array([0])])
for i in range(1000):
    env.step(action_init)
    env.render()
    time.sleep(0.1)
