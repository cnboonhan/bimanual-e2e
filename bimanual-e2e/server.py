import os
import math
import numpy as np
from pathlib import Path
from vuer import Vuer, VuerSession
from vuer.schemas import Hands
from asyncio import sleep
from yourdfpy import URDF
import pyroki as pk
import _solve_ik_with_multiple_targets
from scipy.spatial.transform import Rotation as R
import viser
from viser.extras import ViserUrdf
from dex_retargeting.retargeting_config import RetargetingConfig
from dex_retargeting.seq_retarget import SeqRetargeting
import zmq
import json
import threading
import time


############################################################################################################################### 
# calibration configs
############################################################################################################################### 
script_path = os.path.dirname(os.path.abspath(__name__))
# urdf_path = f"{script_path}/bimanual-e2e/../SO-ARM100/Simulation/SO101/so101_new_calib.urdf"
urdf_path = f"{script_path}/bimanual-e2e/../Wiki-GRx-Models/GRX/GR2/GR2v2.1.1/urdf/GR2v2.1.1.urdf"
left_hand_urdf_path = f"{script_path}/bimanual-e2e/../Wiki-GRx-Models/Dexterous_hand/fourier_hand_6dof/urdf/fourier_left_hand_6dof.urdf"
right_hand_urdf_path = f"{script_path}/bimanual-e2e/../Wiki-GRx-Models/Dexterous_hand/fourier_hand_6dof/urdf/fourier_right_hand_6dof.urdf"
left_hand_init_position = (0.2, 0.0, 0.0)
left_hand_init_quat = (0, 0, 0, 1)
right_hand_init_position = (0.2, 0.0, 0.0)
right_hand_init_quat = (0, 0, 0, 1)  # Flipped orientation (180° around Y-axis)
def position_remap(position):
    return np.array([-position[2], -position[0], position[1]])

###############################################################################################################################
# Dex-retargeting setup
###############################################################################################################################
left_retargeting_config = RetargetingConfig.from_dict({
    "type": "vector",
    "urdf_path": left_hand_urdf_path,
    "wrist_link_name": "L_hand_base_link",
    "target_origin_link_names": [ "L_hand_base_link", "L_hand_base_link", "L_hand_base_link", "L_hand_base_link", "L_hand_base_link" ],
    "target_task_link_names": [ "L_thumb_tip_link",  "L_index_tip_link", "L_middle_tip_link", "L_ring_tip_link", "L_pinky_tip_link" ],
    "target_link_human_indices": [ [ 0, 0, 0, 0, 0 ], [ 4, 8, 12, 16, 20 ] ],
    "target_joint_names": [
        "L_index_proximal_joint",
        "L_middle_proximal_joint", 
        "L_pinky_proximal_joint",
        "L_ring_proximal_joint",
        "L_thumb_proximal_yaw_joint",
        "L_thumb_proximal_pitch_joint"
    ]
})
right_retargeting_config = RetargetingConfig.from_dict({
    "type": "vector",
    "urdf_path": right_hand_urdf_path,
    "wrist_link_name": "R_hand_base_link",
    "target_origin_link_names": [ "R_hand_base_link", "R_hand_base_link", "R_hand_base_link", "R_hand_base_link", "R_hand_base_link" ],
    "target_task_link_names": [ "R_thumb_tip_link",  "R_index_tip_link", "R_middle_tip_link", "R_ring_tip_link", "R_pinky_tip_link" ],
    "target_link_human_indices": [ [ 0, 0, 0, 0, 0 ], [ 4, 8, 12, 16, 20 ] ],
    "target_joint_names": [
        "R_index_proximal_joint",
        "R_middle_proximal_joint",
        "R_pinky_proximal_joint", 
        "R_ring_proximal_joint",
        "R_thumb_proximal_yaw_joint",
        "R_thumb_proximal_pitch_joint"
    ]
})
left_retargeting = left_retargeting_config.build()
right_retargeting = right_retargeting_config.build()

def extract_and_retarget_joint_states(retargeting_obj, hand_data):
    if hand_data is None or len(hand_data) < 400:
        return None
    
    # Extract base position (wrist position at index 0)
    base_landmark_data = hand_data[0:16]
    base_transform_matrix = np.array(base_landmark_data).reshape(4, 4, order='F')
    base_position = base_transform_matrix[:3, 3]
    
    hand_positions = []
    fingertip_indices = [4, 9, 14, 19, 24]
    
    for finger_idx in fingertip_indices:
        landmark_start = finger_idx * 16
        landmark_data = hand_data[landmark_start:landmark_start + 16]
        
        transform_matrix = np.array(landmark_data).reshape(4, 4, order='F')
        position = transform_matrix[:3, 3]
        
        # Subtract base position from fingertip position
        relative_position = position - base_position
        hand_positions.append(relative_position)
    
    # # Convert to numpy array with shape expected by dex_retargeting
    hand_positions = np.array(hand_positions)
    
    try:
        # Call the retargeting function
        retargeted_joints = retargeting_obj.retarget(hand_positions)
        
        return {
            "joint_positions": retargeted_joints,
            "success": True
        }
    except Exception as e:
        print(f"Retargeting failed: {e}")
        return {
            "joint_positions": None,
            "success": False,
            "error": str(e)
        }

############################################################################################################################### 
# Helpers And Init
############################################################################################################################### 
def transform_matrix_to_pose(matrix_data):
    transform_matrix = np.array(matrix_data).reshape(4, 4, order='F')
    position = transform_matrix[:3, 3]
    rotation_matrix = transform_matrix[:3, :3]
     
    rotation = R.from_matrix(rotation_matrix)
    quaternion_xyzw = rotation.as_quat()  # [x, y, z, w]
    quaternion = np.array([quaternion_xyzw[3], quaternion_xyzw[0], quaternion_xyzw[1], quaternion_xyzw[2]])
    return position_remap(position), quaternion


# Vuer Setup
app = Vuer(host="0.0.0.0", cert=f"{script_path}/bimanual-e2e/public.crt", key=f"{script_path}/bimanual-e2e/private.key")

# Pyroki Setup
target_link_names = ["left_end_effector_link", "right_end_effector_link"]
urdf = URDF.load(urdf_path)                                                             
robot = pk.Robot.from_urdf(urdf)    
left_hand_state = {}
right_hand_state = {}
left_hand_ik_solution = []

# State storage for PyZMQ server
current_state = {
    "joint_names": [],
    "joint_angles": [],
    "left_hand_joint_names": left_retargeting_config.target_joint_names,
    "right_hand_joint_names": right_retargeting_config.target_joint_names,
    "left_retargeting_results": None,
    "right_retargeting_results": None
}

# PyZMQ server setup
def zmq_server():
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:8081")
    print("PyZMQ server started on port 8081")
    
    while True:
        try:
            # Wait for request (blocking)
            message = socket.recv()
            print(f"Received request: {message}")
            
            # Convert numpy arrays to lists for JSON serialization
            def convert_numpy_to_list(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_numpy_to_list(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_to_list(item) for item in obj]
                else:
                    return obj
            
            # Send current state as JSON
            state_json = json.dumps({
                "joint_names": current_state["joint_names"],
                "joint_angles": current_state["joint_angles"].tolist() if isinstance(current_state["joint_angles"], np.ndarray) else current_state["joint_angles"],
                "left_hand_joint_names": current_state["left_hand_joint_names"],
                "right_hand_joint_names": current_state["right_hand_joint_names"],
                "left_retargeting_results": convert_numpy_to_list(current_state["left_retargeting_results"]),
                "right_retargeting_results": convert_numpy_to_list(current_state["right_retargeting_results"])
            })
            socket.send_string(state_json)
        except zmq.ZMQError as e:
            print(f"ZMQ error: {e}")
            # Recreate socket on ZMQ errors
            socket.close()
            socket = context.socket(zmq.REP)
            socket.bind("tcp://*:8081")
            time.sleep(0.1)
        except Exception as e:
            print(f"General error: {e}")
            time.sleep(0.1)

server = viser.ViserServer()
server.scene.add_grid("/ground", width=2, height=2)
urdf_vis = ViserUrdf(server, urdf, root_node_name="/robot")
left_hand_target = server.scene.add_transform_controls(
    "/left_hand_target",
    scale=0.15,
    position=left_hand_init_position,
    wxyz=left_hand_init_quat
)

right_hand_target = server.scene.add_transform_controls(
    "/right_hand_target",
    scale=0.15,
    position=right_hand_init_position,
    wxyz=right_hand_init_quat
)

# Vuer Callbacks
@app.add_handler("HAND_MOVE")
async def handler(event, session):
    if 'left' in event.value.keys():
        left_data = event.value['left']
        if len(left_data) >= 400:
            left_hand_state["data"] = left_data
            left_hand_state['pos'], left_hand_state['quat'] = transform_matrix_to_pose(left_data[10*16:11*16])
            flip_quat = R.from_euler('xyz', [math.pi / 2, 0, -math.pi / 2]).as_quat()  
            left_rot = R.from_quat([left_hand_state['quat'][1], left_hand_state['quat'][2], left_hand_state['quat'][3], left_hand_state['quat'][0]])
            flip_rot = R.from_quat(flip_quat)
            flipped_rot = flip_rot * left_rot
            flipped_quat = flipped_rot.as_quat()
            left_hand_state['quat'] = np.array([flipped_quat[3], flipped_quat[0], flipped_quat[1], flipped_quat[2]])
            # print(f"Left hand - Position: {left_hand_state['pos']}, Quaternion: {left_hand_state['quat']}")
    if 'right' in event.value.keys():
        right_data = event.value['right']
        if len(right_data) >= 400:
            right_hand_state["data"] = right_data
            right_hand_state['pos'], right_hand_state['quat'] = transform_matrix_to_pose(right_data[10*16:11*16])
            flip_quat = R.from_euler('xyz', [-math.pi / 2, 0 , math.pi / 2]).as_quat()  
            right_rot = R.from_quat([right_hand_state['quat'][1], right_hand_state['quat'][2], right_hand_state['quat'][3], right_hand_state['quat'][0]])
            flip_rot = R.from_quat(flip_quat)
            flipped_rot = flip_rot * right_rot
            flipped_quat = flipped_rot.as_quat()
            right_hand_state['quat'] = np.array([flipped_quat[3], flipped_quat[0], flipped_quat[1], flipped_quat[2]])
            # print(f"right hand - Position: {right_hand_state['pos']}, Quaternion: {right_hand_state['quat']}")
 
############################################################################################################################### 
# Main
############################################################################################################################### 
# Start PyZMQ server in background thread
zmq_thread = threading.Thread(target=zmq_server, daemon=True)
zmq_thread.start()

@app.spawn(start=True)
async def main(session: VuerSession):
    session.upsert(
        Hands(
            stream=True,
            key="hands",
        ),
        to="bgChildren",
    )

    calibration = {}
    while True:
        if 'pos' in right_hand_state.keys() and 'quat' in right_hand_state.keys() and 'pos' in left_hand_state.keys() and 'quat' in left_hand_state.keys():
            if not calibration:
                calibration = {
                    "left_pos": left_hand_state['pos'] - left_hand_init_position,
                    "right_pos": right_hand_state['pos'] - right_hand_init_position,
                    "left_quat": left_hand_state['quat'],
                    "right_quat": right_hand_state['quat'],
                }

            left_hand_target.position = left_hand_state['pos'] - calibration['left_pos']
            left_hand_target.wxyz = left_hand_state['quat']
            
            right_hand_target.position = right_hand_state['pos'] - calibration['right_pos']
            right_hand_target.wxyz = right_hand_state['quat']

            ik_solution = _solve_ik_with_multiple_targets.solve_ik_with_multiple_targets(                                                                                                          
                robot=robot,
                target_link_names=target_link_names,
                target_positions=np.array([left_hand_target.position, right_hand_target.position]), 
                target_wxyzs=np.array([left_hand_target.wxyz, right_hand_target.wxyz]),
            )

            # Hacky fix for reversed inputs
            ik_solution[15] = ik_solution[15] * -1
            ik_solution[16] = ik_solution[16] * -1

            # Print joint angles
            joint_names = urdf.actuated_joint_names
            # print(f"Joint angles: {dict(zip(joint_names, ik_solution[:len(joint_names)]))}")
            
            left_retargeting_results = extract_and_retarget_joint_states(left_retargeting, left_hand_state.get("data"))
            right_retargeting_results = extract_and_retarget_joint_states(right_retargeting, right_hand_state.get("data"))

            # Update current state for PyZMQ server
            current_state["joint_names"] = joint_names
            current_state["joint_angles"] = ik_solution[:len(joint_names)]
            current_state["left_retargeting_results"] = left_retargeting_results
            current_state["right_retargeting_results"] = right_retargeting_results

            # print(left_retargeting_results)
            # print(right_retargeting_results)
            
            urdf_vis.update_cfg(ik_solution)

        await sleep(0.01)