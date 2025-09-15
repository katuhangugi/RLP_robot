import gymnasium as gym
import numpy as np
import time

env = gym.make("BaseRobot-v0", render_mode="human")
obs, info = env.reset()

# Step 1: 获取绳子末端的位姿
rope_tip_body = "wire0:B_last"
rope_id = env.sim.model.body_name2id(rope_tip_body)
rope_pos = env.sim.data.body_xpos[rope_id].copy()
rope_quat = np.array([1., 0., 0., 0.])  # 可选默认方向

# Step 2: 拼接末端目标姿态
target_pose = np.concatenate([rope_pos, rope_quat])
controller = env.controller
controller.target_type = controller.TargetType.POSE

# Step 3: 控制器逼近目标
for _ in range(300):
    ctrl = np.zeros(env.sim.model.nu)
    controller.run(target_pose, ctrl)
    env.sim.data.ctrl[:] = ctrl
    env.sim.step()
    env.render()
    time.sleep(0.01)

# Step 4: 控制夹爪闭合（你可根据 actuator 索引调整）
for _ in range(50):
    ctrl = np.zeros(env.sim.model.nu)
    ctrl[-2:] = 1.0  # 夹爪闭合控制
    env.sim.data.ctrl[:] = ctrl
    env.sim.step()
    env.render()
    time.sleep(0.01)

env.close()
