import math
import os
from pathlib import Path # type: ignore
from typing import Optional # type: ignore

import numpy as np
import torch
from gym import spaces
import sys

from omni.isaac.core.tasks import BaseTask
from omni.isaac.core.scenes.scene import Scene
from omni.isaac.core.objects import VisualCuboid

from omni.isaac.core.utils.prims import create_prim
from omni.isaac.core.articulations import ArticulationView

ALOHA_ASSET_PATH = (
    Path.home()
    / ".local/share/ov/pkg/isaac_sim-2022.2.1/standalone_examples/aloha-tdmpc/assets/ALOHA.usd"
).as_posix()


class AlohaTask(BaseTask):
    def __init__(self, 
        name: str,
        n_envs: int = 1,
        offset: Optional[np.ndarray] = None,
    ) -> None:
        self.num_envs = n_envs
        self.env_spacing = 1.5
        
        self.action_space = spaces.Box(low=-1, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=float("inf"), high=float("inf"), shape=(16,), dtype=np.float32)
        
        # wheels
        self._wheel_dof_names = ["left_wheel", "right_wheel"]
        self._num_wheel_dof = len(self._wheel_dof_names)
        self._wheel_dof_indices: list[int]
        self.max_velocity = 5
        self.max_angular_velocity = math.pi * 0.5

        BaseTask.__init__(self, name=name, offset=offset)

    def set_up_scene(self, scene: Scene) -> None:
               
        for scene_id in range(self.num_envs):
            scene_prim_path = f"/World/scene_{scene_id}"
            create_prim(
                prim_path=scene_prim_path,
                position=(0, 0 + 3 * scene_id, 0)
            )
            
            # adding robot
            create_prim(
                prim_path=f"{scene_prim_path}/aloha",
                translation=(0,0,0),
                usd_path=ALOHA_ASSET_PATH
            )
            
            
            # adding target location
            tloc = scene.add(
                VisualCuboid(
                    prim_path=f"{scene_prim_path}/target_location",
                    name=f"target_location_{scene_id}",
                    translation=np.array([1.5, 0.2, 0]),
                    size=0.1,
                    color=np.array([0, 1.0, 0]),
                )
            )
        
        self.robots = ArticulationView(
            prim_paths_expr=f"/World/scene_*/aloha",
            name="aloha_view"
        )

        self.tlocs = ArticulationView(
            prim_paths_expr=f"/World/scene_*/target_location",
            name="tloc_view"
        )
        
        scene.add_default_ground_plane()
        scene.add(self.robots)
    
    def reset(self, env_ids=None):
        self.robots.set_joint_positions(self.default_robot_joint_positions)
        
        from omni.isaac.dynamic_control import _dynamic_control
        dc = _dynamic_control.acquire_dynamic_control_interface()
        for i in range(self.num_envs):
            articulation = dc.get_articulation(f"/World/scene_{i}/aloha")
            root_body = dc.get_articulation_root_body(articulation)
            dc.wake_up_articulation(articulation)
            tf = _dynamic_control.Transform()
            tf.p = (0,3*i,0)
            dc.set_rigid_body_pose(root_body, tf)

        
    def post_reset(self) -> None:
        self._wheel_dof_indices = [
            self.robots.get_dof_index(self._wheel_dof_names[i]) for i in range(self._num_wheel_dof)
        ]
        self.default_robot_joint_positions = self.robots.get_joint_positions()
    
    def get_observations(self) -> dict:
        """
        0-2: platform position
        3-6: platform orientation
        7-9: platform linear velocity
        10-12: platform angular velocity
        13-14: gripper_1 joint positions
        15-20: arm_1 joint positions
        21-26: arm_1 joint velocities
        27-28: gripper_2 joint positions
        29-34: arm_2 joint positions
        35-40: arm_2 joint velocities
        41-43: cube positions
        44-47: cube orientations
        48-50: target location positions
        """
        robot_local_positions, robot_local_orientations = self.robots.get_local_poses()
        dof_linvels = self.robots.get_linear_velocities()
        dof_angvels = self.robots.get_angular_velocities()
        

        tloc_pos, tloc_quat = self.tlocs.get_local_poses()
        
        self.obs = torch.cat(
            [   
                robot_local_positions,
                robot_local_orientations,
                dof_linvels,
                dof_angvels,
                tloc_pos,
            ],
            axis=-1
        )
        return self.obs
    
    def calculate_metrics(self) -> dict:
        robot_position = self.obs[:, :3]
        tloc_pos = self.obs[:, 13:16]
        dist = np.linalg.norm(tloc_pos - robot_position)
        rewards = -dist
        return torch.as_tensor(rewards)

    def is_done(self) -> bool:
        dones = torch.tensor([False] * self.num_envs, dtype=bool)
        return dones

    def pre_physics_step(self, actions):
        """
        0-1: wheel velocities
        2: gripper_1 control (1 to open, 0 to close)
        3-8: 6 arm_1 joint position refs
        9: gripper_2 control
        10-15: 6 arm_2 joint position refs
        """
        actions = torch.as_tensor(actions, dtype=torch.float32)

        # Ensure actions are at least 2-dimensional
        if actions.ndim == 1:
            actions = actions.unsqueeze(0)  # Add batch dimension if it's not present

        # control wheel pairs
        # -------------------
        wheel_vels = actions[:, :2]
        print(wheel_vels)
        wheel_vels = torch.as_tensor(wheel_vels).to(torch.float32)
        wheel_vels = torch.clip(wheel_vels, min=-1.0, max=1.0) * 3
        
        
        self.robots.set_joint_velocities(wheel_vels, joint_indices=self._wheel_dof_indices)
        
