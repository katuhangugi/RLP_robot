import gymnasium as gym
import homestri_ur5e_rl
from homestri_ur5e_rl.input_devices.keyboard_input import (
    KeyboardInput,
)

keyboard_input = KeyboardInput()

env = gym.make("Ur5PickAndPlace-v0", render_mode="human")

observation, info = env.reset(seed=42)
gripper = 255
while True:
    # action = env.action_space.sample()

    action = keyboard_input.get_action()
    # 切换 gripper 状态
    if gripper == 255:
        gripper = 0
    else:
        gripper = 255
    action[-1] = gripper  

    observation, reward, terminated, truncated, info = env.step(action)
    
    pos, quat = env.unwrapped.get_object_pose("object0", type="body")
    print("方块位置：", pos)
    # print("方块姿态（四元数）：", quat)
    gripper_pos, gripper_quat = env.unwrapped.get_gripper_pose()
    print("Gripper 位置:", gripper_pos)
    # print("Gripper 姿态（四元数）:", quat)


    if terminated or truncated:
        observation, info = env.reset()

env.close()
