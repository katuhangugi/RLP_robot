import os
import numpy as np
import open3d as o3d
import json


import rtde_control
import rtde_receive

from scipy.spatial.transform import Rotation as R
import pyrealsense2 as rs

from diffusion_policy_3d.env.realUR5.robotiq_gripper import RobotiqGripper
from diffusion_policy_3d.env.realUR5.rs_api import RSCamera, reset_all_cameras

from PIL import Image

from pytorch3d.ops import sample_farthest_points
import torch

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def normalize_vector(x, eps=1e-6):
    """normalize a vector to unit length"""
    x = np.asarray(x)
    if x.ndim == 1:
        norm = np.linalg.norm(x)
        return np.zeros_like(x) if norm < eps else (x / norm)
    elif x.ndim == 2:
        norm = np.linalg.norm(x, axis=1)  # (N,)
        normalized = np.zeros_like(x)
        normalized[norm > eps] = x[norm > eps] / norm[norm > eps][:, None]
        return normalized

def get_langsam_output(image, sam_model, segmentation_text):

    model = sam_model
    img = Image.fromarray(image, mode="RGB")
    imgs = [img]
    segmentation_texts = [segmentation_text]

    results = model.predict(imgs, segmentation_texts)
    # results: List[Dict[str, Any]] = ['scores', 'labels', 'boxes', 'masks', 'mask_scores']

    return results

def get_gdino_output(gdino_model, image, query):
    '''
    image: PLT Image
    '''
    texts_prompt = [query]
    imgs = [image]
    gdino_results = gdino_model.predict(imgs, texts_prompt, box_threshold=0.3, text_threshold=0.25)
    print(gdino_results)
    all_results = []
    sam_images = []
    sam_boxes = []
    sam_indices = []
    for idx, result in enumerate(gdino_results):
        processed_result = {
            **result,
            "masks": [],
            "mask_scores": [],
        }

        if result["labels"]:
            processed_result["boxes"] = result["boxes"].cpu().numpy()
            processed_result["scores"] = result["scores"].cpu().numpy()
            sam_images.append(np.asarray(imgs[idx]))
            sam_boxes.append(processed_result["boxes"])
            sam_indices.append(idx)

        all_results.append(processed_result)

    # if sam_images:
    #     print(f"Predicting {len(sam_boxes)} masks")
    return sam_boxes[0][0]


class Observation(object):
    """Storage for both visual and low-dimensional observations."""

    def __init__(self,
                 obj_points_sampled: np.ndarray,
                 scene_points_sampled: np.ndarray,
                #  left_shoulder_rgb: np.ndarray,
                #  left_shoulder_depth: np.ndarray,
                #  left_shoulder_point_cloud: np.ndarray,
                #  right_shoulder_rgb: np.ndarray,
                #  right_shoulder_depth: np.ndarray,
                #  right_shoulder_point_cloud: np.ndarray,
                #  right_back_rgb: np.ndarray,
                #  right_back_depth: np.ndarray,
                #  right_back_point_cloud: np.ndarray,
                 front_rgb: np.ndarray,
                 front_depth: np.ndarray,
                 front_point_cloud: np.ndarray,
                #  wrist_rgb: np.ndarray,
                #  wrist_depth: np.ndarray,
                #  wrist_point_cloud: np.ndarray,
                 joint_velocities: np.ndarray,
                 joint_positions: np.ndarray,
                 joint_forces: np.ndarray,
                 gripper_open: float,
                 gripper_pose: np.ndarray):
        self.obj_points_sampled = obj_points_sampled
        self.scene_points_sampled = scene_points_sampled
        # self.left_shoulder_rgb = left_shoulder_rgb
        # self.left_shoulder_depth = left_shoulder_depth
        # self.left_shoulder_point_cloud = left_shoulder_point_cloud
        # self.right_shoulder_rgb = right_shoulder_rgb
        # self.right_shoulder_depth = right_shoulder_depth
        # self.right_shoulder_point_cloud = right_shoulder_point_cloud
        # self.right_back_rgb = right_back_rgb
        # self.right_back_depth = right_back_depth
        # self.right_back_point_cloud = right_back_point_cloud
        self.front_rgb = front_rgb
        self.front_depth = front_depth
        self.front_point_cloud = front_point_cloud
        # self.wrist_rgb = wrist_rgb
        # self.wrist_depth = wrist_depth
        # self.wrist_point_cloud = wrist_point_cloud
        self.joint_velocities = joint_velocities
        self.joint_positions = joint_positions
        self.joint_forces = joint_forces
        self.gripper_open = gripper_open
        self.gripper_pose = gripper_pose

    def get_low_dim_data(self) -> np.ndarray:
        """Gets a 1D array of all the low-dimensional observations.

        :return: 1D array of observations.
        """
        low_dim_data = [
            self.joint_velocities,
            self.joint_positions,
            self.joint_forces,
            np.array([self.gripper_open]),
            self.gripper_pose
        ]
        return np.concatenate(low_dim_data)

class VoxPoserUR5():
    def __init__(self, sam_model=None, predictor=None, visualizer=None):
        """
        Initializes the ur5 environment.

        Args:
            sam_model: LangSAM model for object detection, required.
            predictor: SAM2 realtime predictor, required.
            visualizer: Visualization interface, optional.
        """

        # 定义UR5初始关节配置
        self.reset_joints = np.array([-1.5393455664264124, -1.9977265797057093, -1.5625704526901245, -1.1234382551959534, 1.59672212600708, 0.022049665451049805])
        self.control_hz = 10  # 控制频率（默认10Hz）
        
        # 连接 UR5
        self.robot_ip = "192.168.11.101"  # 替换为UR5 IP
        self.rtde_r = rtde_receive.RTDEReceiveInterface(self.robot_ip)
        self.rtde_c = rtde_control.RTDEControlInterface(self.robot_ip)
        print(f"{bcolors.OKGREEN}Connected to UR5 at {self.robot_ip}{bcolors.ENDC}")

        # 初始化夹爪
        self.gripper = RobotiqGripper()
        self.gripper.connect(hostname=self.robot_ip, port=63352)
        print(f"{bcolors.OKGREEN}Robotiq 2F-85 gripper connected{bcolors.ENDC}")

        # 初始化摄像头
        self.cameras = {}
        self.camera_extrinsics = {}  # 外参，定义在标定中
        reset_all_cameras()

        self.camera_configs = {
            # 'left_shoulder': '112322077904',
            # 'right_shoulder': '112422070124',
            'front': '139522077436',
            # 'right_back': '112322073643',
            # 'wrist': '419122270180'
        }       

        for _, serial in self.camera_configs.items():
            self.cameras[serial] = RSCamera(serial)

        self.camera_extrinsics = {
            # 各摄像头的外参矩阵
            # gripper
            '419122270180': np.array([[ 0.99979101,  0.01529234, -0.01356789,  0.00883835],
                        [-0.00831244,  0.91041887,  0.41360415, -0.07873888],
                        [ 0.01867744, -0.41340492,  0.91035571,  0.034132  ],
                        [ 0.        ,  0.        ,  0.        ,  1.        ]]),
            
            # 右中
            '112422070124': np.array([
                [ 4.97305696e-03,  4.36813062e-01, -8.99538558e-01,  7.18726943e-01],
                        [ 9.99915691e-01, -1.29624040e-02, -7.66514905e-04, -7.48347939e-01],
                        [-1.19950059e-02, -8.99458907e-01, -4.36840698e-01,  4.27625982e-01],
                        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]
            ]),
            # 正前方俯视
            '139522077436': np.array([
                [ 0.99927238, -0.00810981,  0.03726843,  0.04025599],
                        [-0.02527053, -0.87265678,  0.48767976, -1.0496097 ],
                        [ 0.02856756, -0.48826671, -0.87222675,  0.7213888 ],
                        [ 0.0,         0.0,         0.0,         1.0]
            ]),
            # 右后
            '112322073643': np.array([
                [-0.78291366,  0.09945271, -0.61412976,  0.58774551],
                        [ 0.6194908,   0.2154605,  -0.75485623, -0.32911013],
                        [ 0.05724821, -0.97143498, -0.23029701,  0.15750787],
                        [ 0.0,         0.0,         0.0,         1.0]
            ]),
            # 左中
            '112322077904': np.array([
                [-0.01480894, -0.5119841,   0.85886726, -0.57227202],
                        [-0.99904627,  0.0428631,   0.00832539, -0.64184573],
                        [-0.04107618, -0.85792484, -0.51213056,  0.42736898],
                        [ 0.0,         0.0,         0.0,         1.0]
            ])
        }

        # to do: update workspace bounds
        self.workspace_bounds_min = np.array([-0.55,-1,-0.2])
        self.workspace_bounds_max = np.array([0.7,-0.3,0.7])
        self.visualizer = visualizer
        if self.visualizer is not None:
            self.visualizer.update_bounds(self.workspace_bounds_min, self.workspace_bounds_max)
        self.camera_names = ['front', 'left_shoulder', 'right_shoulder', 'right_back', 'wrist']
        # self.camera_names = ['front', 'left_shoulder', 'right_shoulder', 'wrist']
        # calculate lookat vector for all cameras (for normal estimation)
        forward_vector = np.array([0, 0, 1])
        self.lookat_vectors = {}

        # 根据相机的校准矩阵计算每个相机的 LookAt 向量
        for cam_name, cam_id in zip(self.camera_names, self.camera_extrinsics.keys()):
            extrinsics = self.camera_extrinsics[cam_id]  # 获取相机的外参矩阵
            lookat = extrinsics[:3, :3] @ forward_vector  # 计算 LookAt 向量（旋转部分应用到前向向量）
            self.lookat_vectors[cam_name] = normalize_vector(lookat)  # 归一化 LookAt 向量
        
        print(f"{bcolors.OKGREEN}UR5 environment initialized{bcolors.ENDC}")

        # 初始化 LangSAM 模型
        assert sam_model is not None, "LangSAM model must be provided"
        self.sam_model = sam_model
        print(f"{bcolors.OKGREEN}LangSAM model loaded{bcolors.ENDC}")

        # 初始化预测器
        assert predictor is not None, "SAM2 realtime predictor must be provided"
        self.predictor = predictor
        print(f"{bcolors.OKGREEN}SAM2 realtime predictor loaded{bcolors.ENDC}")

        # 初始化gripper偏移
        self.gripper_offset = np.array([0.0, 0.0, 0.173])

        # 初始化观测值
        self.reset()
        
    def move_to_initial_position(self):
        """
        Moves the robot to the initial position and opens the gripper.
        """
        self.rtde_c.moveJ(self.reset_joints, speed=0.5, acceleration=0.3)
        self.gripper.move(0, 255, 1)


    def get_3d_obs_by_name(self, query_name):
        """
        Retrieves 3D point cloud observations and normals of an object by its name.

        Args:
            query_name (str): The name of the object to query.

        Returns:
            tuple: A tuple containing object points and object normals.
        """
        # gather points and masks from all cameras
        points, normals, scores = [], [], []
        for cam in self.camera_names:
            # 从最新的观测数据中获取每个相机的点云，并调整其形状为(-1, 3)
            point_cloud = getattr(self.latest_obs, f"{cam}_point_cloud").reshape(-1, 3)
            rgb_img = getattr(self.latest_obs, f"{cam}_rgb")

            # Use LangSAM to get masks
            results = get_langsam_output(rgb_img, self.sam_model, query_name)
            if len(results[0]['scores']) > 0 :
                score = results[0]['scores'][0] # 获取最高置信度
                if score > 0.35:  # 设置最低置信度阈值
                    mask = results[0]['masks'][0].reshape(-1)
                else:
                    mask = np.zeros(rgb_img.shape[:2], dtype=bool).reshape(-1)
            else:
                score = 0  # 没有有效置信度时，默认置信度为 0
                mask = np.zeros(rgb_img.shape[:2], dtype=bool).reshape(-1)

            # estimate normals using o3d
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(point_cloud)
            pcd.estimate_normals()
            cam_normals = np.asarray(pcd.normals)

            # Adjust normal vectors with look-at vector
            flip_indices = np.dot(cam_normals, self.lookat_vectors[cam]) > 0
            cam_normals[flip_indices] *= -1

            scores.append(score)
            points.append(point_cloud[mask > 0])
            normals.append(cam_normals[mask > 0])

        # # 将所有相机的点和法线合并
        # points = np.concatenate(points, axis=0)
        # normals = np.concatenate(normals, axis=0)
        # 获取置信度最高的两个视角的索引
        top_two_indices = np.argsort(scores)[-2:]  # 获取最高分的两个索引
        points = np.concatenate([points[i] for i in top_two_indices], axis=0)
        normals = np.concatenate([normals[i] for i in top_two_indices], axis=0)

        if len(points) == 0:
            print(f"Object {query_name} not found in the scene")
            return None, None

        # 密度过滤去除孤立点
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.normals = o3d.utility.Vector3dVector(normals)

        # # 使用密度过滤
        # cl, ind = pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=1.5)
        # filtered_pcd = pcd.select_by_index(ind)

        # 使用半径过滤
        cl, ind = pcd.remove_radius_outlier(nb_points=30, radius=0.05)
        filtered_pcd = pcd.select_by_index(ind)

        # 转换为 NumPy 数组
        points = np.asarray(filtered_pcd.points)
        normals = np.asarray(filtered_pcd.normals)

        # only keep points within workspace
        chosen_idx_x = (points[:, 0] > self.workspace_bounds_min[0]) & (points[:, 0] < self.workspace_bounds_max[0])
        chosen_idx_y = (points[:, 1] > self.workspace_bounds_min[1]) & (points[:, 1] < self.workspace_bounds_max[1])
        chosen_idx_z = (points[:, 2] > self.workspace_bounds_min[2]) & (points[:, 2] < self.workspace_bounds_max[2])
        points = points[(chosen_idx_x & chosen_idx_y & chosen_idx_z)]
        normals = normals[(chosen_idx_x & chosen_idx_y & chosen_idx_z)]
        
        # voxel downsample using o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.normals = o3d.utility.Vector3dVector(normals)
        pcd_downsampled = pcd.voxel_down_sample(voxel_size=0.001)

        obj_points = np.asarray(pcd_downsampled.points)
        obj_normals = np.asarray(pcd_downsampled.normals)
        return obj_points, obj_normals

    def get_scene_3d_obs(self, ignore_robot=False):
        """
        Retrieves the entire scene's 3D point cloud observations and colors.

        Args:
            ignore_robot (bool): Whether to ignore points corresponding to the robot.

        Returns:
            tuple: A tuple containing scene points and colors.
        """
        points, colors = [], []
        robot_masks = [] if ignore_robot else None

        for cam in self.camera_names:
            point_cloud = getattr(self.latest_obs, f"{cam}_point_cloud").reshape(-1, 3)
            rgb_img = getattr(self.latest_obs, f"{cam}_rgb")
            rgb = rgb_img.reshape(-1, 3)
            if ignore_robot:
                results = get_langsam_output(rgb_img, self.sam_model, "ur5")
                if len(results[0]['scores']) > 0 and results[0]['scores'][0] > 0.35:
                    mask_robot = results[0]['masks'][0].reshape(-1)
                else:
                    mask_robot = np.zeros(rgb_img.shape[:2], dtype=bool).reshape(-1)
                robot_masks.append(mask_robot)

            points.append(point_cloud)
            colors.append(rgb/ 255.0)
            
        points = np.concatenate(points, axis=0)
        colors = np.concatenate(colors, axis=0)

        if ignore_robot:
            robot_masks = np.concatenate(robot_masks, axis=0).astype(bool)
            points = points[~robot_masks]
            colors = colors[~robot_masks]

        # only keep points within workspace
        chosen_idx_x = (points[:, 0] > self.workspace_bounds_min[0]) & (points[:, 0] < self.workspace_bounds_max[0])
        chosen_idx_y = (points[:, 1] > self.workspace_bounds_min[1]) & (points[:, 1] < self.workspace_bounds_max[1])
        chosen_idx_z = (points[:, 2] > self.workspace_bounds_min[2]) & (points[:, 2] < self.workspace_bounds_max[2])
        points = points[(chosen_idx_x & chosen_idx_y & chosen_idx_z)]
        colors = colors[(chosen_idx_x & chosen_idx_y & chosen_idx_z)]

        # voxel downsample using o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        pcd_downsampled = pcd.voxel_down_sample(voxel_size=0.001)
        points = np.asarray(pcd_downsampled.points)
        colors = (np.asarray(pcd_downsampled.colors) * 255).astype(np.uint8)

        return points, colors

    def reset(self):
        """
        Resets the environment and the task. Also updates the visualizer.
        reset observation, action.

        Returns:
            tuple: A tuple containing task descriptions and initial observations.
        """
        """
        重置UR5机械臂至初始状态。
        """
        self.move_to_initial_position()
        self.gripper.auto_calibrate()
        print(f"{bcolors.OKGREEN}Robotiq 2F-85 gripper calibrated{bcolors.ENDC}")
        self._reset_task_variables()
        obs = self.get_observation()
        descriptions = self.get_task_descriptions()
        obs = self._process_obs(obs)
        self.init_obs = obs
        self.latest_obs = obs
        self._update_visualizer()
        
        # reset observation, action sequences
        self.obs_seq = []
        self.next_obs_seq = []
        self.action_seq = []
        self.subtask_info = []

        return descriptions, obs
    
    # todo: update this function
    def get_task_descriptions(self):
        descriptions = [
                            'put the sponge in the blue plate',
                            'drop the sponge into the blue plate',
                            'pick up the sponge and place it in the blue plate',
                            'throw the sponge into the blue plate',
                            'chuck the sponge into the blue plate',
                        ]
        return descriptions
    
    def get_object_names(self):
        return ['sponge', 'blue plate']

    # todo: update this function
    def get_observation(self):
        """
        获取当前观测值。

        :return: Observation 对象。
        """
        # 获取摄像头数据
        # left_shoulder_rgb, left_shoulder_depth, left_shoulder_pcd = self.get_camera_data(self.camera_configs['left_shoulder'])
        # right_shoulder_rgb, right_shoulder_depth, right_shoulder_pcd = self.get_camera_data(self.camera_configs['right_shoulder'])
        # right_back_rgb, right_back_depth, right_back_pcd = self.get_camera_data(self.camera_configs['right_back'])
        front_rgb, front_depth, front_pcd = self.get_camera_data(self.camera_configs['front'])
        # wrist_rgb, wrist_depth, wrist_pcd = self.get_camera_data(self.camera_configs['wrist'])

        # 获取关节状态
        joint_positions = self.rtde_r.getActualQ()
        joint_velocities = self.rtde_r.getActualQd()
        joint_forces = self.rtde_r.getTargetMoment()

        # 获取末端状态
        tcp_pose = self.rtde_r.getActualTCPPose()
        tcp_position = tcp_pose[:3]
        # 这一步的四元数是xyzw格式
        tcp_quaternion = R.from_rotvec(tcp_pose[3:]).as_quat()

        # 获取夹爪状态
        gripper_open = 1.0 if self.gripper.is_open() else 0.0

        frame = Image.fromarray(front_rgb, mode="RGB")
        width, height = frame.size
        # 初始化观测时记录掩码
        if self.init_obs is None:
            self.predictor.load_first_frame(frame)
            ann_frame_idx = 0  # the frame index we interact with

            for obj_idx, obj_name in enumerate(self.get_object_names()):
                sam_box = get_gdino_output(self.sam_model.gdino, frame, obj_name)
                if sam_box is None:
                    continue
                _, out_obj_ids, out_mask_logits = self.predictor.add_new_prompt(
                    frame_idx=ann_frame_idx,
                    obj_id=obj_idx,
                    bbox=sam_box,
                )
        else:
            _, out_mask_logits = self.predictor.track(frame)
         
        out_masks = (out_mask_logits > 0.0).cpu().numpy()  # shape: (N, 1, H, W)
        # 将 N 个对象的 mask 合并成一个整 mask，任意一个对象为 True 即为 True
        combined_mask = np.any(out_masks, axis=0)  # shape: (1, H, W)
        obj_mask = combined_mask[0].reshape(-1)
        obj_points = front_pcd[obj_mask]
        scene_mask = ~obj_mask
        scene_points = front_pcd[scene_mask]

        target_obj_points=1024
        target_scene_points=1024

        if obj_points.shape[0] > target_obj_points:
            # Perform farthest point sampling for object points
            obj_points = torch.from_numpy(obj_points).cuda()
            sampled_obj_points, _ = sample_farthest_points(
                points=obj_points.unsqueeze(0), K=target_obj_points
            )
            obj_points = sampled_obj_points.squeeze(0).cpu().numpy()
        elif obj_points.shape[0] < target_obj_points:
            # Pad with zeros if not enough object points
            padding = np.zeros((target_obj_points - obj_points.shape[0], 3))
            obj_points = np.concatenate([obj_points, padding], axis=0)

        if scene_points.shape[0] > target_scene_points:
            # Perform farthest point sampling for scene points
            scene_points = torch.from_numpy(scene_points).cuda()
            sampled_scene_points, _ = sample_farthest_points(
                points=scene_points.unsqueeze(0), K=target_scene_points
            )
            scene_points = sampled_scene_points.squeeze(0).cpu().numpy()
        elif scene_points.shape[0] < target_scene_points:
            # Pad with zeros if not enough scene points
            padding = np.zeros((target_scene_points - scene_points.shape[0], 3))
            scene_points = np.concatenate([scene_points, padding], axis=0)

        # 创建 Observation 对象
        observation = Observation(
            obj_points_sampled=obj_points,
            scene_points_sampled=scene_points,
            # left_shoulder_rgb=left_shoulder_rgb,
            # left_shoulder_depth=left_shoulder_depth,
            # left_shoulder_point_cloud=left_shoulder_pcd,
            # right_shoulder_rgb=right_shoulder_rgb,
            # right_shoulder_depth=right_shoulder_depth,
            # right_shoulder_point_cloud=right_shoulder_pcd,
            # right_back_rgb=right_back_rgb,
            # right_back_depth=right_back_depth,
            # right_back_point_cloud=right_back_pcd,
            front_rgb=front_rgb,
            front_depth=front_depth,
            front_point_cloud=front_pcd,
            # wrist_rgb=wrist_rgb,
            # wrist_depth=wrist_depth,
            # wrist_point_cloud=wrist_pcd,
            joint_velocities=np.array(joint_velocities),
            joint_positions=np.array(joint_positions),
            joint_forces=np.array(joint_forces),
            gripper_open=gripper_open,
            gripper_pose=np.concatenate([tcp_position, tcp_quaternion])
        )

        return observation

    def get_camera_data(self, serial):
        """
        获取单个摄像头的 RGB、深度和点云数据。

        :param serial: 摄像头序列号。
        :return: (RGB, 深度, 点云) 三元组。
        """
        camera = self.cameras[serial]
        rgb, depth, points = camera.get_camera_data()

        # 转换到世界坐标系
        extrinsics = self.camera_extrinsics[serial]  # 形状 (4, 4)

        # 使用外参矩阵进行转换
        # R = extrinsics[:3, :3] (旋转矩阵), T = extrinsics[:3, 3] (平移向量)
        points_world = (extrinsics[:3, :3] @ points.T + extrinsics[:3, 3:4]).T  # 形状 (n, 3)

        # # 将颜色与世界坐标拼接
        # point_cloud_world_with_color = np.hstack((points_world, colors))  # 形状 (n, 6)

        # return rgb, depth,point_cloud_world_with_color
        return rgb, depth, points_world

    # todo: control the gripper
    def update_ur5(self, action):
        """
        控制UR5运动.
        action: (x, y, z, qx, qy, qz, qw, gripper_state)
        """
        # 转换四元数为旋转向量
        quaternion = action[3:7]
        rotvec = R.from_quat(quaternion).as_rotvec()

        # 构造目标位姿，加上夹爪偏移
        tcp_position = action[:3] + self.gripper_offset
        target_pose = np.concatenate((tcp_position, rotvec))  # 拼接位置和姿态
        
        # 计算逆运动学解
        joint_positions = self.rtde_c.getInverseKinematics(target_pose)
        
        if joint_positions:
            # 控制机械臂移动
            self.rtde_c.moveJ(joint_positions, speed=0.5, acceleration=0.3)
        else:
            print("Inverse kinematics solution not found for target pose:", target_pose)
            return

        # 控制夹爪状态
        gripper_state = action[-1]
        if gripper_state == 1:
            self.gripper.move(0, 255, 1)  # 打开夹爪
        else:
            self.gripper.move(255, 255, 1)  # 关闭夹爪

    def apply_action(self, action):
        """
        执行机械臂动作指令。

        Args:
            action (np.array): 动作指令。

        Returns:
            当前机械臂状态。
        """
        # print(f"Applying action: {action}")
        self.obs_seq.append(self.latest_obs)

        # 对action的四元数格式进行转换
        action = self._process_action(action)

        # 控制UR5机械臂运动
        self.update_ur5(action)

        # 获取当前新的观测obs（从UR5状态与传感器中获取）并将xyzw转换为wxyz
        obs = self.get_observation()
        obs = self._process_obs(obs)

        # 记录最新的状态
        self.latest_obs = obs
        self.latest_action = action

        self.next_obs_seq.append(obs)
        self.action_seq.append(action)

        # 如果需要可视化更新
        self._update_visualizer()

        return obs

    def move_to_pose(self, pose, speed=None):
        """
        Moves the robot arm to a specific pose.

        Args:
            pose: The target pose.
            speed: The speed at which to move the arm. Currently not implemented.

        Returns:
            tuple: A tuple containing the latest observations.
        """
        if self.latest_action is None:
            action = np.concatenate([pose, [self.init_obs.gripper_open]])
        else:
            action = np.concatenate([pose, [self.latest_action[-1]]])
        return self.apply_action(action)
    
    def open_gripper(self):
        """
        Opens the gripper of the robot.
        """
        action = np.concatenate([self.latest_obs.gripper_pose, [1.0]])
        return self.apply_action(action)

    def close_gripper(self):
        """
        Closes the gripper of the robot.
        """
        action = np.concatenate([self.latest_obs.gripper_pose, [0.0]])
        return self.apply_action(action)

    def set_gripper_state(self, gripper_state):
        """
        Sets the state of the gripper.

        Args:
            gripper_state: The target state for the gripper.

        Returns:
            tuple: A tuple containing the latest observations.
        """
        action = np.concatenate([self.latest_obs.gripper_pose, [gripper_state]])
        return self.apply_action(action)

    def reset_to_default_pose(self):
        """
        Resets the robot arm to its default pose.

        Returns:
            tuple: A tuple containing the latest observations.
        """
        if self.latest_action is None:
            action = np.concatenate([self.init_obs.gripper_pose, [self.init_obs.gripper_open]])
        else:
            action = np.concatenate([self.init_obs.gripper_pose, [self.latest_action[-1]]])
        self.subtask_info.append('reset_to_default_pose')
        return self.apply_action(action)

    def get_ee_pose(self):
        assert self.latest_obs is not None, "Please reset the environment first"
        return self.latest_obs.gripper_pose

    def get_ee_pos(self):
        return self.get_ee_pose()[:3] - self.gripper_offset

    def get_ee_quat(self):
        return self.get_ee_pose()[3:]

    def get_last_gripper_action(self):
        """
        Returns the last gripper action.

        Returns:
            float: The last gripper action.
        """
        if self.latest_action is not None:
            return self.latest_action[-1]
        else:
            return self.init_obs.gripper_open

    def _reset_task_variables(self):
        """
        Resets variables related to the current task in the environment.

        Note: This function is generally called internally.
        """
        self.init_obs = None
        self.latest_obs = None
        self.latest_action = None

        # reset observation, action sequences
        self.obs_seq = []
        self.next_obs_seq = []
        self.action_seq = []
        self.subtask_info = []
   
    def _update_visualizer(self):
        """
        Updates the scene in the visualizer with the latest observations.

        Note: This function is generally called internally.
        """
        if self.visualizer is not None:
            points, colors = self.get_scene_3d_obs(ignore_robot=False)
            self.visualizer.update_scene_points(points, colors)
    
    def _process_obs(self, obs):
        """
        Processes the observations, specifically converts quaternion format from xyzw to wxyz.

        Args:
            obs: The observation to process.

        Returns:
            The processed observation.
        """
        quat_xyzw = obs.gripper_pose[3:]
        quat_wxyz = np.concatenate([quat_xyzw[-1:], quat_xyzw[:-1]])
        obs.gripper_pose[3:] = quat_wxyz
        return obs

    def _process_action(self, action):
        """
        Processes the action, specifically converts quaternion format from wxyz to xyzw.

        Args:
            action: The action to process.

        Returns:
            The processed action.
        """
        quat_wxyz = action[3:7]
        quat_xyzw = np.concatenate([quat_wxyz[1:], quat_wxyz[:1]])
        action[3:7] = quat_xyzw
        return action
    
    def disconnect(self):
        """
        Disconnects the UR5 robot and gripper.
        """
        self.gripper.disconnect()
        self.rtde_r.disconnect()
        self.rtde_c.disconnect()

    def __del__(self):
        self.disconnect()
        for _, camera in self.cameras.items():
            camera.close()