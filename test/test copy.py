# import gym
# env = gym.make("UR5ePickAndPlace-v1")  # 或者 FetchPickAndPlace-v1
# print("R5ePickAndPlace-v1:",env.action_space)  # 输出动作空间

# env = gym.make("FetchPickAndPlace-v1")  # 或者 FetchPickAndPlace-v1
# print("FetchPickAndPlace-v1:",env.action_space)  # 输出动作空间

import gym
env = gym.make("UR5ePickAndPlace-v1")
obs = env.reset()
print("UR5e 初始位置:", obs['observation'][:3])  # 检查 x, y, z 位置
