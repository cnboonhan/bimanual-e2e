# conda activate env_isaaclab

import torch
from isaaclab.app import AppLauncher

app_launcher = AppLauncher(enable_cameras=True)

import omni.usd # type: ignore
import omni.physx as physx # type: ignore
import omni.kit.commands as cmd # type: ignore
from omni.isaac.dynamic_control import _dynamic_control as dynamic_control # type: ignore
from omni.isaac.dynamic_control._dynamic_control import Transform # type: ignore
from pxr import Gf, Sdf, UsdLux, UsdGeom, UsdPhysics # type: ignore

from isaaclab.scene import InteractiveScene 
import isaaclab.sim as sim_utils
from assets import TeleoperationSceneCfg
import zmq
import json
import numpy as np

# ZMQ client setup
context = zmq.Context()
socket = context.socket(zmq.REQ)
socket.connect("tcp://localhost:8081")

def get_robot_state_from_zmq():
    """Get robot state from ZMQ server"""
    try:
        socket.send_string("get_state")
        response = socket.recv_string()
        state = json.loads(response)
        return state
    except Exception as e:
        print(f"ZMQ client error: {e}")
        return None

def run_sim(sim: sim_utils.SimulationContext, scene: InteractiveScene, teleop=None, steps: int = 500, reset: bool = True):
    if reset:
        sim.reset()
        sim_init_actions(scene)
    for i in range(steps):
        sim_dt = sim.get_physics_dt()
        sim_time = 0.0
        sim_loop_actions(scene, teleop, sim)
        scene.write_data_to_sim()
        sim.step()
        sim_time += sim_dt
        scene.update(sim_dt)
        print(f"{i}")

def sim_init_actions(scene: InteractiveScene):
    root_state = scene["robot"].data.default_root_state.clone()
    scene["robot"].write_root_pose_to_sim(root_state[:, :7])
    scene["robot"].write_root_velocity_to_sim(root_state[:, 7:])
    scene["robot"].write_joint_state_to_sim(
        scene["robot"].data.default_joint_pos.clone(), 
        scene["robot"].data.default_joint_vel.clone()
    )

def get_normalized_obs(articulation):
    obs = articulation.data.joint_pos.clone()
    for joint in articulation.data.joint_names:
        idx = articulation.data.joint_names.index(joint)
        min_val = articulation.data.joint_pos_limits[0][idx][0]
        max_val = articulation.data.joint_pos_limits[0][idx][1]
        obs[0][idx] = (obs[0][idx] - min_val) / (max_val - min_val)
    return obs

def map_zmq_joints_to_sim(state, sim_joint_names, scene):
    """Map ZMQ joint data to simulation joint order"""
    # Initialize with default joint positions to lock waist joints
    joint_targets = scene["robot"].data.default_joint_pos.clone().cpu().numpy().flatten()
    
    if not state:
        return joint_targets
    
    # Map robot body joints (ignore waist joints - keep them rigid)
    waist_keywords = ["waist", "torso", "spine", "body", "base"]
    if state.get("joint_names") and state.get("joint_angles"):
        for i, zmq_joint_name in enumerate(state["joint_names"]):
            # Skip any joint that might be waist-related
            is_waist_joint = any(keyword in zmq_joint_name.lower() for keyword in waist_keywords)
            if not is_waist_joint and zmq_joint_name in sim_joint_names:
                sim_idx = sim_joint_names.index(zmq_joint_name)
                joint_targets[sim_idx] = state["joint_angles"][i]
                print(f"Updated joint {zmq_joint_name} at index {sim_idx}")
            elif is_waist_joint:
                print(f"Skipping waist joint: {zmq_joint_name}")
    
    # Map left hand joints
    if (state.get("left_retargeting_results") and 
        state["left_retargeting_results"]["success"] and
        state.get("left_hand_joint_names")):
        
        left_joints = state["left_retargeting_results"]["joint_positions"]
        for i, zmq_joint_name in enumerate(state["left_hand_joint_names"]):
            if zmq_joint_name in sim_joint_names:
                sim_idx = sim_joint_names.index(zmq_joint_name)
                joint_targets[sim_idx] = left_joints[i]
    
    # Map right hand joints
    if (state.get("right_retargeting_results") and 
        state["right_retargeting_results"]["success"] and
        state.get("right_hand_joint_names")):
        
        right_joints = state["right_retargeting_results"]["joint_positions"]
        for i, zmq_joint_name in enumerate(state["right_hand_joint_names"]):
            if zmq_joint_name in sim_joint_names:
                sim_idx = sim_joint_names.index(zmq_joint_name)
                joint_targets[sim_idx] = right_joints[i]
    
    return joint_targets

def sim_loop_actions(scene: InteractiveScene, teleop, sim):
    # Get robot state from ZMQ server
    state = get_robot_state_from_zmq()
    
    # Get simulation joint names
    sim_joint_names = scene["robot"].data.joint_names
    print(f"Simulation expects {len(sim_joint_names)} joints: {sim_joint_names}")
    
    if state:
        print(f"ZMQ robot joints: {len(state.get('joint_names', []))}")
        print(f"ZMQ left hand joints: {len(state.get('left_hand_joint_names', []))}")  
        print(f"ZMQ right hand joints: {len(state.get('right_hand_joint_names', []))}")
        
        # Map joints properly
        joint_targets = map_zmq_joints_to_sim(state, sim_joint_names, scene)
        targets = torch.tensor(joint_targets, dtype=torch.float32, device="cuda").unsqueeze(0)
        
        # Set joint positions
        scene["robot"].set_joint_position_target(targets)
        obs = get_normalized_obs(scene["robot"])
        print(f"Targets shape: {targets.shape}, Obs shape: {obs.shape}")

sim_cfg = sim_utils.SimulationCfg(dt=0.005, device="cuda")
sim = sim_utils.SimulationContext(sim_cfg)
scene_cfg = TeleoperationSceneCfg(num_envs=1, env_spacing=2.0)
scene = InteractiveScene(scene_cfg)

while True:
    run_sim(sim, scene, steps=900, reset=True)