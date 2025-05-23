import mujoco
import mujoco.viewer
import numpy as np
import asyncio
import json
import torch
import cv2
import aiohttp
import math
from typing import List, Dict, Any, Tuple, Optional, Set
from ultralytics import YOLO

# ================= CONFIGURATION =================
CONFIG = {
    'api_url': 'https://api.deepseek.com/v1/chat/completions',
    'api_key': 'sk-8735e5bcf05844c9bb8604e380da3c91',  # Replace with your actual API key
    'joint_limits': {
        'shoulder_pan': {'min': -3.14158, 'max': 3.14158, 'default_speed': 0.3},
        'shoulder_lift': {'min': -3.14158, 'max': 3.14158, 'default_speed': 0.2},
        'elbow_flex': {'min': -3.14158, 'max': 3.14158, 'default_speed': 0.2},
        'wrist_flex': {'min': -3.14158, 'max': 3.14158, 'default_speed': 0.15},
        'wrist_roll': {'min': -3.14158, 'max': 3.14158, 'default_speed': 0.15},
        'gripper': {'min': -2.45, 'max': 0.032, 'default_speed': 0.1},
    },
    'camera': {'width': 640, 'height': 480, 'fps': 30},
    'workspace': {'x_range': (-0.3, 0.3), 'y_range': (0, 0.4), 'z_range': (0.05, 0.3)},
    'movement': {'max_step': 0.01, 'min_steps': 5, 'approach_threshold': 0.02},  # 2cm threshold
    'gripper': {'open_pos': 0.032, 'close_pos': {'mouse': -0.8, 'default': -1.0}}  # Less aggressive grip
}

# Expanded calibration examples matching XML joint names
EXAMPLES = [
    {"coords": (0.1, 0.2, 0.05), "joints": {
        "shoulder_pan": 0.785, "shoulder_lift": 1.047, 
        "elbow_flex": 1.571, "wrist_flex": -0.785, 
        "wrist_roll": 0.0, "gripper": -1.0}},
    {"coords": (-0.15, 0.25, 0.08), "joints": {
        "shoulder_pan": -0.524, "shoulder_lift": 0.785, 
        "elbow_flex": 1.309, "wrist_flex": -0.524, 
        "wrist_roll": 0.0, "gripper": -1.2}},
    {"coords": (0.2, 0.15, 0.1), "joints": {
        "shoulder_pan": 1.047, "shoulder_lift": 0.524, 
        "elbow_flex": 1.047, "wrist_flex": -1.047, 
        "wrist_roll": 0.0, "gripper": -0.8}},
]

# Enhanced system prompt with XML joint names
SYSTEM_PROMPT = """You control a 6-DOF robotic arm. Generate movement commands in this exact JSON format:
{
  "action_type": "pick_up",
  "steps": [
    {"joint": "shoulder_pan", "position": 0.0, "speed": 0.2},
    {"joint": "shoulder_lift", "position": 1.0, "speed": 0.2},
    {"joint": "gripper", "position": -1.0, "speed": 0.1}
  ]
}

Requirements:
1. Target the end_effector body to reach (x,y,z) coordinates
2. Joint limits (from XML file):
   - shoulder_pan: [-3.14, 3.14] (rotation around Z)
   - shoulder_lift: [-3.14, 3.14] (rotation around X)
   - elbow_flex: [-3.14, 3.14] (rotation around X)
   - wrist_flex: [-3.14, 3.14] (rotation around Y)
   - wrist_roll: [-3.14, 3.14] (rotation around Z)
   - gripper: [-2.45, 0.032] (special gripper control)
3. Max step: 0.03 radians between movements
4. Use smooth cosine interpolation
5. Approach from z+0.03, then descend to exact z, grip, and lift to z+0.05

Example solutions for reference:
""" + "\n".join(
    f"- {ex['coords']} → " + ", ".join(f"{j}:{v:.3f}" for j,v in ex["joints"].items())
    for ex in EXAMPLES
)

class SimulatedVisionSystem:
    def __init__(self, model):
        self.model = model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.yolo_model = YOLO("yolov8n.pt").to(self.device)
        self.distance_sensor = SimulatedDistanceSensor()

    def get_mouse_position(self) -> Tuple[float, float, float]:
        """Get precise mouse position from simulation"""
        mouse_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "mouse")
        return tuple(self.model.body_pos[mouse_body])

    async def detect_objects(self) -> List[Dict[str, Any]]:
        """Detect objects with accurate coordinates"""
        mouse_pos = self.get_mouse_position()
        
        # Create visualization
        img = np.zeros((CONFIG['camera']['height'], CONFIG['camera']['width'], 3), dtype=np.uint8)
        x_center = int((mouse_pos[0] + 0.3) / 0.6 * CONFIG['camera']['width'])
        y_center = int(mouse_pos[1] / 0.4 * CONFIG['camera']['height'])
        box_size = 50
        
        cv2.rectangle(img, 
                     (max(0, x_center-box_size//2), max(0, y_center-box_size//2)),
                     (min(CONFIG['camera']['width'], x_center+box_size//2), 
                      min(CONFIG['camera']['height'], y_center+box_size//2)),
                     (0,255,0), 2)
        cv2.imshow("Detection", img)
        cv2.waitKey(1)
        
        return [{
            "world_coords": list(mouse_pos),
            "object_type": "mouse",
            "confidence": 0.95
        }]

class SimulatedDistanceSensor:
    def __init__(self):
        self.last_distance = 0.15

    def get_distance(self, end_effector_pos: np.ndarray, object_pos: np.ndarray) -> float:
        """Get real distance between gripper and object"""
        return np.linalg.norm(end_effector_pos - object_pos)

async def get_arm_command_from_deepseek(data: Dict[str, Any], available_joints: Set[str]) -> Dict[str, Any]:
    """Get validated movement commands from DeepSeek API"""
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps({
                "object_type": data["object_type"],
                "world_coords": [float(x) for x in data["world_coords"]],
                "current_arm_position": {k: float(v) for k,v in data["current_arm_position"].items()},
                "nearby_examples": sorted(EXAMPLES, 
                    key=lambda e: math.dist(data["world_coords"], e["coords"]))[:3]
            }, indent=2)}
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.1,
        "max_tokens": 1000
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                CONFIG['api_url'],
                json=payload,
                headers={"Authorization": f"Bearer {CONFIG['api_key']}"},
                timeout=aiohttp.ClientTimeout(total=10)
            ) as resp:
                
                if resp.status != 200:
                    error_text = await resp.text()
                    print(f"API Error {resp.status}: {error_text}")
                    raise ValueError(f"API error {resp.status}")
                
                response = await resp.json()
                content = response['choices'][0]['message']['content']
                
                try:
                    command = json.loads(content)
                except json.JSONDecodeError as e:
                    print(f"JSON parse error: {e}")
                    raise ValueError("Invalid JSON response")
                
                # Validate command structure
                required_keys = {'action_type', 'steps'}
                if not all(k in command for k in required_keys):
                    raise ValueError(f"Missing required keys: {required_keys - set(command.keys())}")
                
                if not isinstance(command['steps'], list):
                    raise ValueError("'steps' must be a list")
                
                # Filter steps to only include available joints
                command['steps'] = [step for step in command['steps'] if step['joint'] in available_joints]
                
                # Validate each movement step
                valid_joints = set(CONFIG['joint_limits'].keys())
                for step in command['steps']:
                    if 'joint' not in step or 'position' not in step:
                        raise ValueError("Each step must have 'joint' and 'position'")
                    if step['joint'] not in valid_joints:
                        raise ValueError(f"Invalid joint: {step['joint']}")
                    
                    # Clamp position to joint limits
                    joint = step['joint'].strip() 
                    step['joint'] = joint
                    limits = CONFIG['joint_limits'][joint]
                    step['position'] = np.clip(
                        step['position'],
                        limits['min'],
                        limits['max']
                    )
                
                return command
    
    except Exception as e:
        print(f"API processing error: {str(e)}")
        raise

class SimulatedRoboticArm:
    def __init__(self, model_path="robot_arm.xml"):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.viewer = None
        self._emergency_flag = False

        print("Available bodies:")
        for i in range(self.model.nbody):
            print(f"{i}: {self.model.body(i).name}")


        # Create joint name to qpos index mapping
        self.joint_indices = {}
        self.current_positions = {}
        
        for i in range(self.model.njnt):
            joint_name = self.model.joint(i).name
            if joint_name:  # Only add named joints
                self.joint_indices[joint_name] = self.model.joint(i).qposadr[0]
                self.current_positions[joint_name] = self.data.qpos[self.joint_indices[joint_name]]
        
        print(f"Initialized with joints: {list(self.joint_indices.keys())}")
        
        # Initialize distance sensor
        self.distance_sensor = SimulatedDistanceSensor()

    def get_arm_state(self) -> Dict[str, float]:
        """Get current joint positions"""
        return self.current_positions.copy()

    def get_end_effector_position(self) -> np.ndarray:
        """Get end effector position in world coordinates"""
        ee_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "end_effector")
        ee_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "end_effector")
        pos = self.data.geom_xpos[ee_geom_id].copy()

        return self.data.geom_xpos[ee_geom_id].copy()


    async def execute_deepseek_command(self, command: Dict[str, Any]) -> bool:
        """Execute validated movement command with enhanced approach logic"""
        print(f"Executing {command['action_type']} action")
        # Step 0: open the gripper before approaching object
        open_gripper_pos = CONFIG['gripper']['open_pos']
        if 'gripper' in self.joint_indices:
            print("Opening gripper before approach...")
            for i in range(10):
                ratio = 0.5 * (1 - math.cos(math.pi * (i / 10)))
                pos = self.current_positions['gripper'] + (open_gripper_pos - self.current_positions['gripper']) * ratio
                self.data.qpos[self.joint_indices['gripper']] = pos
                self.current_positions['gripper'] = pos
                mujoco.mj_step(self.model, self.data)
                if self.viewer:
                    self.viewer.sync()
                await asyncio.sleep(0.01)


        for step in command['steps']:
            if self._emergency_flag:
                break
                
            joint = step['joint']
            target = step['position']
            speed = step.get('speed', CONFIG['joint_limits'][joint]['default_speed'])
            
            current = self.current_positions[joint]
            distance = abs(target - current)
            steps = max(
                CONFIG['movement']['min_steps'],
                int(distance / CONFIG['movement']['max_step'])
            )
            
            # Special handling for gripper closing
            if joint == 'gripper' and target < 0:
                mouse_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "mouse")
                if mouse_body_id == -1:
                    print("Mouse body not found in model!")
                    return False

                mouse_pos = self.data.body(mouse_body_id).xpos
                ee_pos = self.get_end_effector_position()
                distance = self.distance_sensor.get_distance(ee_pos, mouse_pos)

                if distance > 0.03:  # 3cm threshold
                    print(f"Adjusting position - current distance: {distance:.3f}m")
                    
                    # Calculate direction to mouse
                    direction = mouse_pos - ee_pos
                    direction = direction / np.linalg.norm(direction) * 0.02  # 2cm step
                    
                    # Adjust joints to move closer
                    self.current_positions["shoulder_pan"] += direction[1] * 0.5
                    self.current_positions["shoulder_lift"] -= direction[2] * 0.3
                    self.current_positions["elbow_flex"] += direction[2] * 0.5
                    
                    # Apply the adjustments
                    for j in ["shoulder_pan", "shoulder_lift", "elbow_flex"]:
                        self.data.qpos[self.joint_indices[j]] = self.current_positions[j]
                    
                    mujoco.mj_step(self.model, self.data)
                    if self.viewer:
                        self.viewer.sync()
                    await asyncio.sleep(0.5)
                    continue  # Retry this step after adjustment

            # Execute smooth movement
            for i in range(1, steps + 1):
                ratio = 0.5 * (1 - math.cos(math.pi * (i / steps)))
                pos = current + (target - current) * ratio
                
                self.data.qpos[self.joint_indices[joint]] = pos
                self.current_positions[joint] = pos
                
                mujoco.mj_step(self.model, self.data)
                if self.viewer:
                    self.viewer.sync()
                await asyncio.sleep(0.01)
        
        return await self.verify_grip()

    async def verify_grip(self) -> bool:
        """Check if object was successfully gripped with more precise checks"""
        mouse_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "mouse")
        if mouse_body_id == -1:
            return False

        mouse_pos = self.data.body(mouse_body_id).xpos
        gripper_pos = self.current_positions['gripper']
        ee_pos = self.get_end_effector_position()
        
        # Calculate distance between gripper and mouse
        distance = np.linalg.norm(mouse_pos - ee_pos)
        
        # Check if mouse has moved significantly from its original position
        original_mouse_pos = np.array([0.0, 0.25, 0.05])
        mouse_moved = np.linalg.norm(mouse_pos - original_mouse_pos) > 0.01
        
        # Successful grip conditions:
        # 1. Gripper is closed enough
        # 2. Mouse is close to end effector
        # 3. Mouse has moved from its original position
        success = (
            gripper_pos < -0.5 and 
            distance < 0.02 and 
            mouse_moved
        )
        
        print(f"Grip verification - Distance: {distance:.3f}m, Gripper pos: {gripper_pos:.2f}, Moved: {mouse_moved}")
        print("Grip successful!" if success else "Grip failed")
        return success

    async def retry_grip(self, command: Dict[str, Any], retries: int = 2) -> bool:
        """Retry grip operation with small adjustments"""
        for attempt in range(retries):
            print(f"Grip attempt {attempt + 1}/{retries}")
            
            # Make small adjustment to approach position
            self.current_positions['elbow_flex'] -= 0.02  # Move slightly down
            self.data.qpos[self.joint_indices['elbow_flex']] = self.current_positions['elbow_flex']
            mujoco.mj_step(self.model, self.data)
            
            # Execute command
            success = await self.execute_deepseek_command(command)
            if success:
                return True
                
            await asyncio.sleep(0.5)
            
        return False

    def render(self):
        """Initialize viewer if not already running"""
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.cam.distance = 1.0
            self.viewer.cam.lookat[:] = [0, 0.2, 0.1]
            self.viewer.cam.elevation = -20
            self.viewer.cam.azimuth = 180

    def __del__(self):
        """Clean up resources"""
        if hasattr(self, 'viewer') and self.viewer is not None:
            self.viewer.close()
        cv2.destroyAllWindows()

def create_fallback_command(world_coords: List[float], current_pos: Dict[str, float]) -> Dict[str, Any]:
    """Generate simple IK solution when API fails"""
    x, y, z = world_coords
    
    # Basic inverse kinematics for first 3 joints
    shoulder_pan = math.atan2(y, x)
    shoulder_lift = math.pi/2 - math.atan2(z, math.hypot(x, y))
    elbow_flex = math.pi/2 - shoulder_lift
    
    return {
        "action_type": "pick_up",
        "steps": [
            # Approach phase - get near the mouse
            {"joint": "shoulder_pan", "position": shoulder_pan, "speed": 0.2},
            {"joint": "shoulder_lift", "position": shoulder_lift, "speed": 0.2},
            {"joint": "elbow_flex", "position": elbow_flex, "speed": 0.2},
            
            # Fine positioning - get closer
            {"joint": "elbow_flex", "position": elbow_flex + 0.15, "speed": 0.1},  # Move down
            {"joint": "wrist_flex", "position": -0.5, "speed": 0.1},  # Tilt wrist
            
            # Grip sequence
            {"joint": "gripper", "position": CONFIG['gripper']['open_pos'], "speed": 0.1},  # Open gripper
            {"joint": "elbow_flex", "position": elbow_flex + 0.2, "speed": 0.05},  # Final approach
            {"joint": "gripper", "position": CONFIG['gripper']['close_pos']['mouse'], "speed": 0.05},  # Close gripper
            
            # Lift
            {"joint": "elbow_flex", "position": elbow_flex + 0.1, "speed": 0.1},  # Lift slightly
        ],
        "target_coords": world_coords
    }

async def main_control_loop():
    """Main robot control loop"""
    arm = SimulatedRoboticArm()
    vision = SimulatedVisionSystem(arm.model)
    arm.render()
    
    try:
        while True:
            try:
                # Detect objects
                detections = await vision.detect_objects()
                if not detections:
                    await asyncio.sleep(0.1)
                    continue
                
                detection = detections[0]
                coords = detection['world_coords']
                print(f"\nDetected object at: {coords}")
                
                # Get movement command
                try:
                    cmd = await get_arm_command_from_deepseek({
                        "object_type": detection["object_type"],
                        "world_coords": coords,
                        "current_arm_position": arm.get_arm_state()
                    }, set(arm.joint_indices.keys()))
                    cmd["target_coords"] = coords
                except Exception as e:
                    print(f"Error getting API command: {str(e)}")
                    print("Using fallback command...")
                    cmd = create_fallback_command(coords, arm.get_arm_state())
                
                # Execute command
                success = await arm.execute_deepseek_command(cmd)
                if not success and cmd['action_type'] == 'pick_up':
                    print("Initial grip failed, retrying...")
                    success = await arm.retry_grip(cmd)
                
                print("Movement completed successfully" if success else "Movement failed")
                await asyncio.sleep(2)
                
            except Exception as e:
                print(f"Control loop error: {str(e)}")
                await asyncio.sleep(1)
                
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")

if __name__ == "__main__":
    asyncio.run(main_control_loop())