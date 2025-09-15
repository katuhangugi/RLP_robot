from typing import Union, Optional, Optional, Tuple
import numpy as np
import gym
from gym.spaces import Box, Dict
from diffusion_policy_3d.env.realUR5 import VoxPoserUR5
from termcolor import cprint

from diffusion_policy_3d.model.text_encoder.clip_encoders import ClipLanguageConditioned

CLIP_TEXT_ENCODER = ClipLanguageConditioned()

class RealUR5DiffusionWrapper():
    """An 3d-Diffusion env wrapper for real UR5 env.
    Ponit clouds of background and objects.
    Add task embedding to observation.
    """
    
    def __init__(self,
            env: VoxPoserUR5,
            seed: Optional[int]=None,
            cam_name: str = 'front', # 'front', 'left_shoulder', 'right_shoulder', 'overhead', 'wrist'
            num_points  = 1024,
            point_sampling_method = 'fps',
            text_encoder=CLIP_TEXT_ENCODER,
        ):
        self.env = env
        if seed is not None:
            np.random.seed(seed)

        self.cam_name = cam_name
        self.num_points = num_points
        self.point_sampling_method = point_sampling_method
        cprint(f"[RLBenchEnvDiffusionWrapper] sampling {self.num_points} points from point cloud using {self.point_sampling_method}", 'green')
        descriptions, obs = self.env.reset()
        self.obs = obs
        self.descriptions = descriptions
        self.instruction = np.random.choice(descriptions)

        self.obj_names = self.env.get_object_names()

        self.text_encoder = text_encoder

        self.obs_sensor_dim = 26

        self.action_space = Box(low=-np.inf, high=np.inf, shape=(8,))
        self.observation_space = Dict({
            'image': Box(
                low=0,
                high=255,
                shape=(480, 640, 3),
                dtype=np.uint8
            ),
            'depth': Box(
                low=0,
                high=1,
                shape=(480, 640),
                dtype=np.float32
            ),
            'agent_pos': Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.obs_sensor_dim,),
                dtype=np.float32
            ),
            'point_cloud_bg': Box( 
                low=-np.inf,
                high=np.inf,
                shape=(self.num_points , 3),
                dtype=np.float32
            ),
            'point_cloud_obj': Box( 
                low=-np.inf,
                high=np.inf,
                shape=(self.num_points , 3),
                dtype=np.float32
            ),
            'task_embed': Box(
                low=-np.inf,
                high=np.inf,
                shape=(512,),
                dtype=np.float32
            ),
        })   

    def seed(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
            return [seed]
        else:
            return [None]

    def reset(self):
        descriptions, obs = self.env.reset()
        self.obs = obs
        self.instruction = np.random.choice(descriptions)
        return self._get_obs()
    
        
    def render(self, mode='rgb_array'):
        if mode == 'rgb_array':
            # 返回 camera 视角的 RGB 图像
            return getattr(self.obs, f'{self.cam_name}_rgb')
        elif mode == 'depth_array':
            return getattr(self.obs, f'{self.cam_name}_depth')
    
    def step(self, action) -> Tuple[dict, float, bool, dict]:
        action = self._process_action(action)
        # print(f"action: {action}")
        obs = self.env.apply_action(action)
        self.obs = obs
        obs_dict = self._get_obs()

        return obs_dict, 0, False, {}

    def _process_action(self, action: np.ndarray) -> np.ndarray:
        for i in range(3):  # 设置动作的边界
            action[i] = np.clip(action[i], self.env.workspace_bounds_min[i], self.env.workspace_bounds_max[i])
        # 处理四元数
        action[3:7] = self._normalize_quaternion(action[3:7])
        # 处理抓手的开闭状态
        action[7] = self._process_gripper(action[7])
        return action
    
    def _normalize_quaternion(self, quaternion_xyzw):
        # 将 xyzw 转换为 wxyz
        quaternion_wxyz = np.roll(quaternion_xyzw, shift=1)
        # 标准化四元数以确保其为单位四元数
        norm = np.linalg.norm(quaternion_wxyz)
        if norm == 0:
            raise ValueError("Cannot normalize a zero norm quaternion")
        quaternion_normalized = quaternion_wxyz / norm
        return quaternion_normalized
    
    def _process_gripper(self, gripper):
        return 1.0 if gripper > 0.5 else 0.0
    
    def _get_obs(self, obs=None):
        if obs is None:
            obs = self.obs
        obs_dict = {}
        obs_dict['image'] = getattr(obs, f'{self.cam_name}_rgb')
        obs_dict['depth'] = getattr(obs, f'{self.cam_name}_depth')

        obs_dict['point_cloud_obj'] = getattr(obs, 'obj_points_sampled')
        obs_dict['point_cloud_bg'] = getattr(obs, 'scene_points_sampled')

        obs_dict['agent_pos'] = obs.get_low_dim_data()
        obs_dict['task_embed'] = self.text_encoder.get_text_feature(self.instruction).detach().numpy()
        return obs_dict
