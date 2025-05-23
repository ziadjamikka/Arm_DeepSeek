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
    'movement': {
        'max_step': 0.01,
        'min_steps': 5,
        'approach_threshold': 0.02  # 2cm threshold
    },
    'gripper': {
        'open_pos': 0.032,
        'close_pos': {'mouse': -0.8, 'default': -1.0}  # Less aggressive grip
    }
}

# Calibration examples
EXAMPLES = [
    {"coords": (0.1, 0.2, 0.05), "joints": {
        "shoulder_pan": 0.785, "shoulder_lift": 1.047, 
        "elbow_flex": 1.571, "wrist_flex": -0.785, 
        "wrist_roll": 0.0, "gripper": -0.8}},
    {"coords": (-0.15, 0.25, 0.08), "joints": {
        "shoulder_pan": -0.524, "shoulder_lift": 0.785, 
        "elbow_flex": 1.309, "wrist_flex": -0.524, 
        "wrist_roll": 0.0, "gripper": -0.8}},
]

# System prompt for API
SYSTEM_PROMPT = """You control a 6-DOF robotic arm. Generate movement commands in this exact JSON format:
{
  "action_type": "pick_up",
  "steps": [
    {"joint": "shoulder_pan", "position": 0.0, "speed": 0.2},
    {"joint": "shoulder_lift", "position": 1.0, "speed": 0.2},
    {"joint": "gripper", "position": -0.8, "speed": 0.1}
  ]
}

Joint limits:
- shoulder_pan: [-3.14, 3.14] (rotation around Z)
- shoulder_lift: [-3.14, 3.14] (rotation around X)
- elbow_flex: [-3.14, 3.14] (rotation around X)
- wrist_flex: [-3.14, 3.14] (rotation around Y)
- wrist_roll: [-3.14, 3.14] (rotation around Z)
- gripper: [-2.45, 0.032] (special gripper control)

Movement requirements:
1. Approach from z+0.01
2. Lift to z+0.02 after grip
3. Max step: 0.03 radians between movements
4. Use smooth cosine interpolation

Example solutions:
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
    
    def get_distance(self) -> float:
        """Get distance with minimal noise"""
        self.last_distance += np.random.uniform(-0.001, 0.001)
        return np.clip(self.last_distance, *CONFIG['workspace']['z_range'])

async def get_arm_command_from_deepseek(data: Dict[str, Any], available_joints: Set[str]) -> Dict[str, Any]:
    """Get movement commands from DeepSeek API"""
    payload = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps({
                    "object_type": data["object_type"],
                    "world_coords": [float(x) for x in data["world_coords"]],
                    "current_arm_position": {k: float(v) for k, v in data["current_arm_position"].items()}
                }, indent=2)}
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.1,
            "max_tokens": 1000
        }


    async with aiohttp.ClientSession() as session:
        async with session.post(
            CONFIG['api_url'],
            json=payload,
            headers={"Authorization": f"Bearer {CONFIG['api_key']}"},
            timeout=aiohttp.ClientTimeout(total=10)
        ) as resp:
            
            if resp.status != 200:
                error_text = await resp.text()
                raise ValueError(f"API error {resp.status}: {error_text}")
            
            response = await resp.json()
            content = response['choices'][0]['message']['content']
            
            try:
                command = json.loads(content)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON response: {e}")

            # Validate command structure
            if not all(k in command for k in ('action_type', 'steps')):
                raise ValueError("Missing required keys in response")
            
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
                joint = step['joint']
                limits = CONFIG['joint_limits'][joint]
                step['position'] = np.clip(
                    step['position'],
                    limits['min'],
                    limits['max']
                )
            
            command['target_coords'] = data["world_coords"]
            return command

class SimulatedRoboticArm:
    def __init__(self, model_path="robot_arm.xml"):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.viewer = None
        self._emergency_flag = False
        
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
        return self.data.body(ee_id).xpos.copy()

    async def execute_deepseek_command(self, command: Dict[str, Any]) -> bool:
        """Execute API movement command with improved approach"""
        print(f"Executing {command['action_type']} action")
        
        # First move all joints EXCEPT gripper
        non_gripper_steps = [s for s in command['steps'] if s['joint'] != 'gripper']
        for step in non_gripper_steps:
            joint = step['joint']
            target = step['position']
            speed = step.get('speed', CONFIG['joint_limits'][joint]['default_speed'])
            
            current = self.current_positions[joint]
            distance = abs(target - current)
            steps = max(
                CONFIG['movement']['min_steps'],
                int(distance / CONFIG['movement']['max_step'])
            )
            
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
        
        # Fine-tune approach if we have target coordinates
        if 'target_coords' in command:
            target_pos = command['target_coords']
            ee_pos = self.get_end_effector_position()
            distance = np.linalg.norm(np.array(ee_pos) - np.array(target_pos))
            
            if distance > CONFIG['movement']['approach_threshold']:
                print(f"Fine-tuning approach (distance: {distance:.3f}m)")
                await self.fine_tune_approach(target_pos)
        
        # Now execute gripper commands if any
        gripper_steps = [s for s in command['steps'] if s['joint'] == 'gripper']
        for step in gripper_steps:
            joint = step['joint']
            target = step['position']
            speed = step.get('speed', CONFIG['joint_limits'][joint]['default_speed'])
            
            current = self.current_positions[joint]
            distance = abs(target - current)
            steps = max(
                CONFIG['movement']['min_steps'],
                int(distance / CONFIG['movement']['max_step'])
            )
            
            for i in range(1, steps + 1):
                ratio = 0.5 * (1 - math.cos(math.pi * (i / steps)))
                pos = current + (target - current) * ratio
                
                self.data.qpos[self.joint_indices[joint]] = pos
                self.current_positions[joint] = pos
                
                mujoco.mj_step(self.model, self.data)
                if self.viewer:
                    self.viewer.sync()
                await asyncio.sleep(0.01)
        
        # Verify grip for pick-up actions
        if command['action_type'] == 'pick_up' and gripper_steps:
            return await self.verify_grip()
        return True

    async def fine_tune_approach(self, target_pos: List[float]) -> bool:
        """Make small adjustments to get closer to target"""
        for _ in range(3):  # Max 3 adjustment attempts
            ee_pos = self.get_end_effector_position()
            error = np.array(target_pos) - np.array(ee_pos)
            
            # Calculate joint adjustments
            adjustments = {
                'shoulder_pan': error[1] * 0.5,
                'shoulder_lift': -error[2] * 0.3,
                'elbow_flex': error[2] * 0.2,
                'wrist_flex': error[0] * 0.1
            }
            
            # Apply adjustments
            for joint, adjustment in adjustments.items():
                if joint in self.joint_indices:
                    current = self.current_positions[joint]
                    new_pos = current + adjustment
                    limits = CONFIG['joint_limits'][joint]
                    new_pos = np.clip(new_pos, limits['min'], limits['max'])
                    
                    self.data.qpos[self.joint_indices[joint]] = new_pos
                    self.current_positions[joint] = new_pos
            
            mujoco.mj_step(self.model, self.data)
            if self.viewer:
                self.viewer.sync()
            
            await asyncio.sleep(0.1)
            
            # Check if we're close enough now
            ee_pos = self.get_end_effector_position()
            distance = np.linalg.norm(np.array(ee_pos) - np.array(target_pos))
            if distance <= CONFIG['movement']['approach_threshold']:
                return True
        
        return False

    async def verify_grip(self) -> bool:
        """Improved grip verification"""
        mouse_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "mouse")
        mouse_pos = self.data.body(mouse_body).xpos.copy()
        gripper_pos = self.current_positions['gripper']
        
        # Check if mouse has moved from initial position
        initial_mouse_pos = np.array([0.0, 0.25, 0.05])
        displacement = np.linalg.norm(mouse_pos - initial_mouse_pos)
        
        # Check if gripper is closed enough
        gripper_closed = gripper_pos < -0.5
        
        # Check if mouse is between gripper pads
        ee_pos = self.get_end_effector_position()
        in_gripper = abs(mouse_pos[0] - ee_pos[0]) < 0.02 and abs(mouse_pos[1] - ee_pos[1]) < 0.02
        
        success = displacement > 0.01 and gripper_closed and in_gripper
        print(f"Grip verification - Displacement: {displacement:.3f}, Closed: {gripper_closed}, In gripper: {in_gripper}")
        return success

    def render(self):
        """Initialize viewer"""
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

async def main_control_loop():
    """Main robot control loop using only API commands"""
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
                
                # Get movement command from API
                cmd = await get_arm_command_from_deepseek({
                    "object_type": detection["object_type"],
                    "world_coords": coords,
                    "current_arm_position": arm.get_arm_state()
                }, set(arm.joint_indices.keys()))
                
                # Execute command
                success = await arm.execute_deepseek_command(cmd)
                if not success and cmd['action_type'] == 'pick_up':
                    print("Initial grip failed, retrying...")
                    success = await arm.execute_deepseek_command(cmd)  # Simple retry
                
                print("Movement completed successfully" if success else "Movement failed")
                await asyncio.sleep(2)
                
            except Exception as e:
                print(f"Control loop error: {str(e)}")
                await asyncio.sleep(1)
                
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")

if __name__ == "__main__":
    asyncio.run(main_control_loop())