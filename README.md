# Arm_DeepSeek
Description:

This project simulates a low-cost 6-DOF robotic arm equipped with AI-driven vision and control capabilities. The robotic arm operates within a Mujoco-based simulation environment, utilizing object detection, distance estimation, and intelligent motion planning to autonomously locate and pick up objects (e.g., a simulated mouse).

Key components of the system include:

YOLOv8 Object Detection: Integrates the Ultralytics YOLO model to identify objects in the simulated environment.

3D Vision System: Tracks object positions in real-world coordinates and visualizes detection results in real-time using OpenCV.

DeepSeek AI Integration: Sends object coordinates and current joint states to the DeepSeek API, which returns optimized movement plans in JSON format.

Motion Execution Engine: Smoothly interpolates joint movements using cosine functions while respecting physical joint limits and safety constraints.

Gripping Logic: Dynamically adjusts the gripper approach using distance measurements and retry strategies to ensure successful object manipulation.

Fallback System: Uses basic inverse kinematics to generate backup movement plans if the AI service is unavailable.

This project showcases how advanced AI techniques can be integrated into a low-cost robotic system to achieve intelligent interaction and autonomy, making it ideal for research, education, and prototyping.
![image](https://github.com/user-attachments/assets/d3adb183-bf8c-484a-b385-1ebe019e845f)

