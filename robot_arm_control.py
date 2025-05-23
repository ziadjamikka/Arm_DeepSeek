import asyncio
import json
import time
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from dynamixel_sdk import PortHandler, PacketHandler
import aiohttp
from typing import Dict, List, Optional, Set, Any, Tuple
import os
import math

# ================= CONFIGURATION =================
CONFIG = {
    'dynamixel_port': 'COM7',
    'baudrate': 1000000,
    'api_url': 'https://api.deepseek.com/v1/chat/completions',
    'api_key': 'sk-8735e5bcf05844c9bb8604e380da3c91', 
    'motors': {
        'joint1': 1,  # Shoulder pan (Z-axis)
        'joint2': 2,   # Shoulder lift (X-axis)
        'joint3': 3,   # Elbow flex (X-axis)
        'joint4': 4,   # Wrist roll (negative Y-axis)
        'joint5': 5    # Gripper (negative Z-axis)
    },
    'joint_limits': {
        'joint1': {'min': 1025, 'max': 3005, 'default_speed': 40, 'rad_per_unit': 0.0015},
        'joint2': {'min': 575, 'max': 1575, 'default_speed': 30, 'rad_per_unit': 0.0015},
        'joint3': {'min': 1813, 'max': 3225, 'default_speed': 30, 'rad_per_unit': 0.0015},
        'joint4': {'min': 808, 'max': 3068, 'default_speed': 20, 'rad_per_unit': 0.0015},
        'joint5': {'min': 1988, 'max': 2989, 'default_speed': 10, 'rad_per_unit': 0.001}
    },
    'camera': {
        'width': 640,
        'height': 480,
        'fps': 30
    },
    'workspace': {
        'x_range': (-0.3, 0.3),
        'y_range': (0, 0.4),
        'z_range': (0.05, 0.3)
    },
    'movement': {
        'max_step': 0.15,
        'approach_height': 0.03,
        'lift_height': 0.05,
        'min_steps': 3
    },
    'gripper': {
        'open_pos': 2989,
        'close_pos': {
            'mouse': 2200,
            'bottle': 2100,
            'cube': 1988,
            'default': 2000
        }
    }
}

PROTOCOL_VERSION = 2.0
ADDR_TORQUE_ENABLE = 64
ADDR_GOAL_POSITION = 116
ADDR_PRESENT_POSITION = 132
ADDR_MOVING_SPEED = 104
DXL_MOVING_STATUS_THRESHOLD = 30
MOVEMENT_TIMEOUT = 7

SYSTEM_PROMPT = """
You are an object-driven robotic arm controller for a low-cost 5-DOF arm. Generate precise joint movements in JSON format with these requirements:

1. Joint Configuration:
- joint1: Shoulder pan (Z-axis, range: 1025-3005 units) 
- joint2: Shoulder lift (X-axis, range: 575-1575 units)
- joint3: Elbow flex (X-axis, range: 1813-3225 units) 
- joint4: Wrist roll (negative Y-axis, range: 808-3068 units)
- joint5: Gripper (negative Z-axis, range: 1988-2989 units)

2. Movement Rules:
- ALWAYS include current_arm_position in calculations
- Use maximum step size of 200 units
- Include smooth acceleration/deceleration
- Maintain all joint limits strictly

3. Required JSON Response Format:
{
  "action_type": "pick_up",
  "steps": [
    {"joint": "joint1", "position": 2000, "speed": 40},
    {"joint": "joint2", "position": 1200, "speed": 30},
    {"joint": "joint3", "position": 2500, "speed": 30},
    {"joint": "joint5", "position": 2989, "duration": 0.5}
  ]
}
"""

# ================= ROBOTIC ARM CLASS =================
class RoboticArm:
    """Robotic arm controller with Dynamixel motors"""
    
    def __init__(self):
        self.port_handler = PortHandler(CONFIG['dynamixel_port'])
        self.packet_handler = PacketHandler(PROTOCOL_VERSION)
        self.active_motors: Set[int] = set()
        self._emergency_flag = False
        self.port_open = False
        self.current_positions = {
            joint: CONFIG['joint_limits'][joint]['min'] 
            for joint in CONFIG['motors']
        }
        
    async def initialize(self):
        """Initialize with retry logic"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                if not self.port_open:
                    if not self.port_handler.openPort():
                        raise ConnectionError("Failed to open port")
                    if not self.port_handler.setBaudRate(CONFIG['baudrate']):
                        raise ConnectionError("Failed to set baud rate")
                    self.port_open = True
                
                for joint, motor_id in CONFIG['motors'].items():
                    self.enable_torque(motor_id)
                    if 'default_speed' in CONFIG['joint_limits'][joint]:
                        self.set_movement_speed(
                            motor_id, 
                            CONFIG['joint_limits'][joint]['default_speed']
                        )
                    self.current_positions[joint] = self.get_current_position(motor_id)
                
                print("Robotic arm initialized successfully")
                return
                
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                print(f"Retrying initialization... ({attempt + 1}/{max_retries})")
                await asyncio.sleep(1)
    
    def enable_torque(self, motor_id: int):
        """Enable torque for a motor"""
        dxl_comm_result, dxl_error = self.packet_handler.write1ByteTxRx(
            self.port_handler, motor_id, ADDR_TORQUE_ENABLE, 1)
        if dxl_comm_result != 0 or dxl_error != 0:
            raise ConnectionError(
                f"Motor {motor_id} torque enable failed: "
                f"Comm result: {dxl_comm_result}, Error: {dxl_error}"
            )
        self.active_motors.add(motor_id)
    
    def disable_torque(self, motor_id: int):
        """Disable torque for a motor"""
        dxl_comm_result, dxl_error = self.packet_handler.write1ByteTxRx(
            self.port_handler, motor_id, ADDR_TORQUE_ENABLE, 0)
        if dxl_comm_result == 0 and dxl_error == 0:
            self.active_motors.discard(motor_id)
    
    def get_current_position(self, motor_id: int) -> int:
        """Read current position of a motor"""
        dxl_present_position, dxl_comm_result, dxl_error = self.packet_handler.read4ByteTxRx(
            self.port_handler, motor_id, ADDR_PRESENT_POSITION)
        if dxl_comm_result != 0 or dxl_error != 0:
            raise ConnectionError(
                f"Failed to read position from motor {motor_id}: "
                f"Comm result: {dxl_comm_result}, Error: {dxl_error}"
            )
        return dxl_present_position
    
    def set_movement_speed(self, motor_id: int, speed: int):
        """Set movement speed for a motor (0-100)"""
        if not 0 <= speed <= 100:
            raise ValueError("Speed must be between 0 and 100")
        
        dxl_comm_result, dxl_error = self.packet_handler.write4ByteTxRx(
            self.port_handler, motor_id, ADDR_MOVING_SPEED, speed)
        if dxl_comm_result != 0 or dxl_error != 0:
            raise ConnectionError(
                f"Failed to set speed for motor {motor_id}: "
                f"Comm result: {dxl_comm_result}, Error: {dxl_error}"
            )
    
    def _validate_position(self, joint: str, position: int) -> bool:
        """Check if position is within safe limits"""
        limits = CONFIG['joint_limits'][joint]
        return limits['min'] <= position <= limits['max']

    def _clamp_position(self, joint: str, position: int) -> int:
        """Clamp position to be within safe limits"""
        limits = CONFIG['joint_limits'][joint]
        return max(limits['min'], min(position, limits['max']))
    
    async def safe_move(self, motor_id: int, position: int, speed: Optional[int] = None):
        """Move motor to position with verification"""
        if self._emergency_flag:
            raise RuntimeError("Emergency stop activated")
            
        if speed is not None:
            self.set_movement_speed(motor_id, speed)
        
        joint = [k for k, v in CONFIG['motors'].items() if v == motor_id][0]
        position = self._clamp_position(joint, position)
        
        dxl_comm_result, dxl_error = self.packet_handler.write4ByteTxRx(
            self.port_handler, motor_id, ADDR_GOAL_POSITION, position)
        if dxl_comm_result != 0 or dxl_error != 0:
            raise ConnectionError(
                f"Failed to set position for motor {motor_id}: "
                f"Comm result: {dxl_comm_result}, Error: {dxl_error}"
            )
        
        self.current_positions[joint] = position
    
        current_pos = self.get_current_position(motor_id)
        timeout = 2 + abs(current_pos - position) / 500
        start_time = time.time()
        while time.time() - start_time < timeout:
            current_pos = self.get_current_position(motor_id)
            if abs(current_pos - position) < DXL_MOVING_STATUS_THRESHOLD:
                return
            await asyncio.sleep(0.1)
        
        raise TimeoutError(
            f"Motor {motor_id} didn't reach target position {position} "
            f"(current: {current_pos}) within {timeout:.2f} seconds"
        )
    
    async def execute_deepseek_command(self, command: Dict[str, Any]):
        """Execute commands received from DeepSeek with smooth movement"""
        print(f"Executing {command['action_type']} action")
        
        for step in command['steps']:
            if self._emergency_flag:
                break

            joint = step['joint']
            motor_id = CONFIG['motors'][joint]
            target_pos = self._clamp_position(joint, step['position'])
            speed = step.get('speed', CONFIG['joint_limits'][joint]['default_speed'])
            
            current_pos = self.current_positions[joint]
            distance = abs(target_pos - current_pos)
            steps = max(CONFIG['movement']['min_steps'],
                        int(distance / 200))  # Using 200 as max step size
            
            for i in range(1, steps + 1):
                ratio = i / steps
                smooth_ratio = ratio**2 * (3 - 2 * ratio)  # Smooth curve
                intermediate_pos = int(current_pos + (target_pos - current_pos) * smooth_ratio)
                
                await self.safe_move(motor_id, intermediate_pos, speed)
                await asyncio.sleep(0.02)  # 50Hz control rate
            
            # Final position verification
            await self.safe_move(motor_id, target_pos, speed)
            
            # Additional duration if specified
            if 'duration' in step:
                await asyncio.sleep(step['duration'])
    
    async def move_to_object(self, world_coords: Tuple[float, float, float], object_type: str):
        """Move robot arm to the detected object's world coordinates and grip it."""
        # Calculate joint positions from world coordinates
        joint_positions = self.calculate_joint_positions(world_coords)
        
        # Generate movement commands for approach
        command = {
            "action_type": "move_to_object",
            "steps": [
                {"joint": "joint1", "position": joint_positions[0], "speed": 40},
                {"joint": "joint2", "position": joint_positions[1], "speed": 30},
                {"joint": "joint3", "position": joint_positions[2], "speed": 30},
                {"joint": "joint4", "position": joint_positions[3], "speed": 20},
                {"joint": "joint5", "position": CONFIG['gripper']['open_pos'], "duration": 0.5}  # Open gripper
            ]
        }
        
        # Execute the approach movement
        await self.execute_deepseek_command(command)
        
        # Add small delay before gripping
        await asyncio.sleep(0.5)
        
        # Generate gripping command based on object type
        grip_pos = CONFIG['gripper']['close_pos'].get(object_type, CONFIG['gripper']['close_pos']['default'])
        grip_command = {
            "action_type": "grip_object",
            "steps": [
                {"joint": "joint5", "position": grip_pos, "duration": 0.5}  # Close gripper
            ]
        }
        
        # Execute the gripping action
        await self.execute_deepseek_command(grip_command)
        
        # Add small delay after gripping
        await asyncio.sleep(0.5)
        
        # Lift the object slightly
        lifted_coords = (world_coords[0], world_coords[1], world_coords[2] + CONFIG['movement']['lift_height'])
        lift_joint_positions = self.calculate_joint_positions(lifted_coords)
        
        lift_command = {
            "action_type": "lift_object",
            "steps": [
                {"joint": "joint2", "position": lift_joint_positions[1], "speed": 20},
                {"joint": "joint3", "position": lift_joint_positions[2], "speed": 20}
            ]
        }
        
        await self.execute_deepseek_command(lift_command)
    
    def calculate_joint_positions(self, world_coords: Tuple[float, float, float]) -> List[int]:
        """Calculate joint positions based on world coordinates (simplified inverse kinematics)"""
        x, y, z = world_coords
        
        # Simplified inverse kinematics for a 5-DOF arm
        # Note: This is a placeholder - you should replace with your actual IK calculations
        joint1_rad = math.atan2(y, x)  # Base rotation
        distance_xy = math.sqrt(x**2 + y**2)
        
        # Arm length parameters (adjust based on your robot's dimensions)
        L1 = 0.1  # Shoulder to elbow length
        L2 = 0.1  # Elbow to wrist length
        
        try:
            # Shoulder and elbow angles
            D = math.sqrt(distance_xy**2 + (z - L1)**2)
            alpha = math.atan2(z - L1, distance_xy)
            beta = math.acos((L1**2 + D**2 - L2**2) / (2 * L1 * D))
            
            joint2_rad = alpha + beta  # Shoulder lift
            joint3_rad = math.acos((L1**2 + L2**2 - D**2) / (2 * L1 * L2)) - math.pi  # Elbow flex
            
            # Wrist angle (simplified to point downward)
            joint4_rad = -joint2_rad - joint3_rad  # Wrist roll
            
            # Convert radians to Dynamixel units
            joint1_pos = int(2048 + joint1_rad / CONFIG['joint_limits']['joint1']['rad_per_unit'])
            joint2_pos = int(1024 + joint2_rad / CONFIG['joint_limits']['joint2']['rad_per_unit'])
            joint3_pos = int(2048 + joint3_rad / CONFIG['joint_limits']['joint3']['rad_per_unit'])
            joint4_pos = int(2048 + joint4_rad / CONFIG['joint_limits']['joint4']['rad_per_unit'])
            joint5_pos = CONFIG['gripper']['open_pos']
            
            return [
                self._clamp_position('joint1', joint1_pos),
                self._clamp_position('joint2', joint2_pos),
                self._clamp_position('joint3', joint3_pos),
                self._clamp_position('joint4', joint4_pos),
                joint5_pos
            ]
            
        except ValueError:
            # If position is unreachable, return current positions
            print("Warning: Target position may be unreachable")
            return [self.current_positions[f'joint{i+1}'] for i in range(5)]
    
    def activate_emergency_stop(self):
        """Immediately stop all movements"""
        self._emergency_flag = True
        for motor_id in list(self.active_motors):
            try:
                current_pos = self.get_current_position(motor_id)
                self.packet_handler.write4ByteTxRx(
                    self.port_handler, motor_id, ADDR_GOAL_POSITION, current_pos)
                self.disable_torque(motor_id)
            except Exception as e:
                print(f"Error during emergency stop for motor {motor_id}: {e}")
    
    def get_arm_state(self) -> Dict[str, int]:
        """Return current positions of all joints"""
        return self.current_positions
    
    def __del__(self):
        """Cleanup on destruction"""
        try:
            for motor_id in list(self.active_motors):
                try:
                    self.disable_torque(motor_id)
                except Exception as e:
                    print(f"Error disabling torque for motor {motor_id}: {e}")
            if self.port_open:
                self.port_handler.closePort()
        except Exception as e:
            print(f"Error during cleanup: {e}")

# ================= VISION SYSTEM =================
class VisionSystem:
    """Vision system with coordinate transformation"""
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = YOLO("yolo11x.pt").to(self.device)
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG['camera']['width'])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG['camera']['height'])
        self.cap.set(cv2.CAP_PROP_FPS, CONFIG['camera']['fps'])
        
        if not self.cap.isOpened():
            raise RuntimeError("Could not open camera")
        
        if os.path.exists('calibration_data.npz'):
            calib_data = np.load('calibration_data.npz')
            self.camera_matrix = calib_data['mtx']
            self.dist_coeffs = calib_data['dist']
        else:
            self.camera_matrix = np.array([
                [923.5, 0, 322.7],
                [0, 925.3, 238.4],
                [0, 0, 1]
            ])
            self.dist_coeffs = np.array([0.112, -0.253, 0.0012, -0.0006, 0.152])
        
        self.transform_matrix = np.array([
            [0.0008, 0, -0.25],
            [0, 0.0009, -0.1],
            [0, 0, 1]
        ])
        self.distance_sensor = DistanceSensor()
    
    def pixel_to_world(self, x: float, y: float) -> Tuple[float, float, float]:
        """
        Convert pixel coordinates to world coordinates with distortion correction.
        Returns (x, y, z) in meters.
        """
        pts = np.array([[[x, y]]], dtype=np.float32)
        undistorted = cv2.undistortPoints(pts, self.camera_matrix, self.dist_coeffs, P=self.camera_matrix)
        x_undist, y_undist = undistorted[0][0]
        world_coords = np.dot(self.transform_matrix, np.array([x_undist, y_undist, 1]))
        world_x, world_y = world_coords[0], world_coords[1]
        world_z = self.distance_sensor.get_distance()
        
        # Clip to workspace limits
        world_x = np.clip(world_x, *CONFIG['workspace']['x_range'])
        world_y = np.clip(world_y, *CONFIG['workspace']['y_range'])
        world_z = np.clip(world_z, *CONFIG['workspace']['z_range'])
        
        return world_x, world_y, world_z
    
    async def detect_objects(self) -> List[Dict[str, Any]]:
        """Perform object detection and return detections"""
        ret, frame = self.cap.read()
        if not ret:
            return []
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.model(frame_rgb)
        
        detections = []
        for result in results:
            if result and hasattr(result, 'boxes') and result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = box.conf[0].item()
                    cls = int(box.cls[0].item())
                    label = self.model.names[cls]
                    
                    # Only process specific objects
                    if label not in ['mouse', 'bottle', 'cube']:
                        continue
                    
                    # Calculate center of bounding box
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    
                    # Convert to world coordinates
                    world_x, world_y, world_z = self.pixel_to_world(center_x, center_y)
                    
                    detections.append({
                        "label": label,
                        "confidence": conf,
                        "pixel_coords": [x1, y1, x2, y2],
                        "world_coords": [world_x, world_y, world_z],
                        "object_type": label
                    })
                    
                    # Draw bounding box and label
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{label} {conf:.2f}", 
                               (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.5, (0, 255, 0), 2)
        
        cv2.imshow("Object Detection", frame)
        cv2.waitKey(1)
        
        # Sort by confidence (highest first)
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        return detections
    
    def __del__(self):
        """Cleanup"""
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()

# ================= DISTANCE SENSOR (SIMULATED) =================
class DistanceSensor:
    """Simulated distance sensor class"""
    def __init__(self):
        self.last_distance = 0.15  # Default value in meters
    
    def get_distance(self) -> float:
        """Get the current distance in meters"""
        # Simulate slight random variations in distance
        self.last_distance += np.random.uniform(-0.01, 0.01)
        self.last_distance = np.clip(self.last_distance, *CONFIG['workspace']['z_range'])
        return self.last_distance

# ================= DEEPSEEK API INTEGRATION =================
async def get_arm_command_from_deepseek(detection_data: Dict[str, Any]) -> Dict[str, Any]:
    """Retrieve movement commands from DeepSeek API"""
    try:
        # Prepare request data
        request_data = {
            "object_type": str(detection_data["object_type"]),
            "world_coords": [float(coord) for coord in detection_data["world_coords"]],
            "current_arm_position": {
                joint: int(pos) 
                for joint, pos in detection_data["current_arm_position"].items()
            }
        }

        payload = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(request_data, indent=2)}
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.1,
            "max_tokens": 1000
        }

        headers = {
            'Authorization': f"Bearer {CONFIG['api_key']}",
            'Content-Type': 'application/json'
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                CONFIG['api_url'],
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=10)
            ) as resp:
                if resp.status != 200:
                    error_detail = await resp.text()
                    raise ValueError(f"API error {resp.status}: {error_detail}")

                data = await resp.json()
                content = data['choices'][0]['message']['content']
                command = json.loads(content)

                # Validate response structure
                if not isinstance(command, dict):
                    raise ValueError("Response is not a JSON object")
                
                if 'action_type' not in command:
                    command['action_type'] = 'pick_up'
                
                if 'steps' not in command or not isinstance(command['steps'], list):
                    command['steps'] = generate_default_movement(
                        request_data['object_type'],
                        request_data['world_coords'],
                        request_data['current_arm_position']
                    )
                
                return command

    except Exception as e:
        print(f"API communication error: {str(e)}")
        # Fallback to default movement
        return {
            "action_type": "pick_up",
            "steps": generate_default_movement(
                detection_data["object_type"],
                detection_data["world_coords"],
                detection_data["current_arm_position"]
            ),
            "error": str(e)
        }

# ================= DEFAULT MOVEMENT GENERATOR =================
def generate_default_movement(object_type: str, world_coords: List[float], current_pos: Dict[str, int]) -> List[Dict[str, Any]]:
    """Generate safe default movement sequence"""
    # Calculate joint positions
    joint_positions = RoboticArm().calculate_joint_positions(world_coords)
    
    # Grip parameters based on object type
    grip_pos = CONFIG['gripper']['close_pos'].get(object_type, CONFIG['gripper']['close_pos']['default'])
    
    return [
        # Approach position
        {"joint": "joint1", "position": joint_positions[0], "speed": 40},
        {"joint": "joint2", "position": joint_positions[1], "speed": 30},
        {"joint": "joint3", "position": joint_positions[2], "speed": 30},
        {"joint": "joint4", "position": joint_positions[3], "speed": 20},
        
        # Open gripper
        {"joint": "joint5", "position": CONFIG['gripper']['open_pos'], "duration": 0.5},
        
        # Close gripper
        {"joint": "joint5", "position": grip_pos, "duration": 0.5},
        
        # Lift object
        {"joint": "joint2", "position": current_pos["joint2"] + 100, "speed": 20},
        {"joint": "joint3", "position": current_pos["joint3"] - 100, "speed": 20}
    ]

# ================= MAIN CONTROL LOOP =================
async def main_control_loop():
    """Main control loop with error handling"""
    arm = RoboticArm()
    vision = VisionSystem()
    
    try:
        await arm.initialize()
        
        while True:
            try:
                # Detect objects
                detections = await vision.detect_objects()
                if not detections:
                    await asyncio.sleep(0.1)
                    continue
                
                # Process the highest-confidence detection
                primary_detection = detections[0]
                world_coords = primary_detection['world_coords']
                
                # Verify distance
                actual_distance = vision.distance_sensor.get_distance()
                if abs(actual_distance - world_coords[2]) > 0.05:
                    world_coords[2] = actual_distance
                
                print(f"Detected {primary_detection['label']} at {world_coords}")
                
                # Option 1: Use DeepSeek API for movement planning
                # command = await get_arm_command_from_deepseek({
                #     "object_type": primary_detection["label"],
                #     "world_coords": world_coords,
                #     "current_arm_position": arm.get_arm_state()
                # })
                # await arm.execute_deepseek_command(command)
                
                # Option 2: Direct movement using inverse kinematics
                await arm.move_to_object(world_coords, primary_detection['object_type'])
                
                print("Movement completed successfully")
                
                await asyncio.sleep(1)
                
            except Exception as e:
                print(f"Control error: {str(e)}")
                await asyncio.sleep(1)
                continue
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        del vision
        del arm
        print("System shutdown complete")

if __name__ == "__main__":
    asyncio.run(main_control_loop())