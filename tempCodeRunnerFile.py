import mujoco
import mujoco.viewer
import numpy as np

class SimulatedRoboticArm:
    def __init__(self, model_path="robot_arm.xml"):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

    def move_joint(self, joint_name, position):
        joint_id = self.model.joint(name=joint_name)
        self.data.qpos[joint_id] = position
        mujoco.mj_forward(self.model, self.data)

    def render(self):
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            while viewer.is_running():
                mujoco.mj_step(self.model, self.data)
                viewer.sync()
