# conda activate env_isaaclab
import requests
import math

from torch import neg

from isaaclab.scene import InteractiveScene 
from isaaclab.assets.rigid_object.rigid_object_cfg import RigidObjectCfg
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg 
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass
from isaaclab_assets.robots.fourier import GR1T2_CFG  
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

@configclass
class TeleoperationSceneCfg(InteractiveSceneCfg):
    """Design the scene with sensors on the robot."""
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    dome_light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)))
    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=(
                f"{ISAAC_NUCLEUS_DIR}/Robots/FourierIntelligence/GR-1/GR1T2_fourier_hand_6dof/GR1T2_fourier_hand_6dof.usd"
            ),
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                retain_accelerations=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1.0,
                kinematic_enabled=False,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.95),
            joint_pos={
                "left_shoulder_pitch_joint" : -1.15,
                "left_shoulder_roll_joint" : 0.4,
                "left_shoulder_yaw_joint" : -1.2,
                "left_wrist_yaw_joint" : 1.0,
                "right_shoulder_pitch_joint" : -1.15,
                "right_shoulder_roll_joint" : -0.4,
                "right_shoulder_yaw_joint" : 1.2,
                "right_wrist_yaw_joint" : -1.0,
            },
            joint_vel={".*": 0.0},
        ),
        actuators=GR1T2_CFG.actuators,
        collision_group=GR1T2_CFG.collision_group,
        debug_vis=GR1T2_CFG.debug_vis,
    )
    camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/head_yaw_link/front_cam",
        update_period=0.005, 
        height=480,
        width=640,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=10.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
        ),
        offset=CameraCfg.OffsetCfg(pos=(0.12, 0.0, 0.0), 
        rot=(0.6699, 0.2377, -0.222, -0.667), convention="opengl"),
    )
    packing_table = AssetBaseCfg(
        prim_path="/World/envs/env_.*/PackingTable",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.561, -0.48, 0.0), rot=(0.71, 0.0, 0.0, -0.704)),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/PackingTable/packing_table.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=False),
        ),
    )
    box_1 = RigidObjectCfg(
        prim_path="/World/envs/env_.*/PackingTable/Box_1",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(-0.30, -0.2, 3), 
            rot=(1.0, 0.0, 0.0, 0.0),  
            lin_vel=(0.0, 0.0, 0.0),  
            ang_vel=(0.0, 0.0, 0.0),  
        ),
        spawn=sim_utils.CuboidCfg(
            size=(0.1, 0.1, 0.1),  
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,  
                kinematic_enabled=False,  
                rigid_body_enabled=True,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=1
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.1,
                rest_offset=0.0,
                collision_enabled=True,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 0.5, 0.0),  # Orange color
                metallic=0.2,
                roughness=0.8,
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="max",
                restitution_combine_mode="min",
                static_friction=10.0,
                dynamic_friction=10.0,
            ),
        ),
    )
    # box_2 = RigidObjectCfg(
    #     prim_path="/World/envs/env_.*/PackingTable/Box_2",
    #     init_state=RigidObjectCfg.InitialStateCfg(
    #         pos=(-0.32, 0.0, 1.2),  
    #         rot=(0.707, 0.0, 0.707, 0.0),  
    #         # rot=(0.0, 0.0, 1.0, 0.0),  
    #         lin_vel=(0.0, 0.0, 0.0),  
    #         ang_vel=(0.0, 0.0, 0.0),  
    #     ),
    #     spawn=sim_utils.CuboidCfg(
    #         size=(0.1, 0.1, 0.1),  # Width, depth, height
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(
    #             rigid_body_enabled=True,
    #             disable_gravity=False,  
    #             kinematic_enabled=False,  
    #         ),
    #         mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
    #         collision_props=sim_utils.CollisionPropertiesCfg(
    #             contact_offset=0.1,
    #             rest_offset=0.0,
    #             collision_enabled=True,
    #         ),
    #         visual_material=sim_utils.PreviewSurfaceCfg(
    #             diffuse_color=(0.0, 0.5, 1.0), 
    #             metallic=0.2,
    #             roughness=0.8,
    #         ),
    #         physics_material=sim_utils.RigidBodyMaterialCfg(
    #             friction_combine_mode="max",
    #             restitution_combine_mode="min",
    #             static_friction=10.0,
    #             dynamic_friction=10.0,
    #         ),
    #     ),
    # )
