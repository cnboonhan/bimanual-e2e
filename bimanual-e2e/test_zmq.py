import zmq
import json

context = zmq.Context()
socket = context.socket(zmq.REQ)
socket.connect("tcp://localhost:8081")

# Send request and receive state
socket.send_string("get_state")
response = socket.recv_string()
state = json.loads(response)

# Access the state data
print("Robot Joint Names:", state["joint_names"])
print("Robot Joint Angles:", state["joint_angles"])
print("Left Hand Joint Names:", state["left_hand_joint_names"]) 
print("Right Hand Joint Names:", state["right_hand_joint_names"])
print("Left Retargeting Results:", state["left_retargeting_results"])
print("Right Retargeting Results:", state["right_retargeting_results"])

# Clean up
socket.close()
context.term()
