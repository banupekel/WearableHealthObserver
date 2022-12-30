import os
import mujoco as mj
import numpy as np

# Load the model XML file
model_path = "humanoid.xml"
model = mj.MjModel.from_xml_path(model_path)

# Create the simulation environment
sim = mj.MjSim(model)

# Set the initial pose of the model to a standing pose
sim.data.qpos[:] = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
sim.data.qvel[:] = np.zeros(model.nv)

# Define the desired walking motion using a set of joint position and velocity targets
num_steps = 10
step_duration = 0.5
pos_targets = np.zeros((num_steps, model.nq))
vel_targets = np.zeros((num_steps, model.nv))
for i in range(num_steps):
    # Set the joint position targets for the left and right legs
    pos_targets[i, model.get_joint_qpos_addr("left_leg_joint")] = np.pi / 4
    pos_targets[i, model.get_joint_qpos_addr("right_leg_joint")] = -np.pi / 4

    # Set the joint velocity targets for the left and right legs
    vel_targets[i, model.get_joint_qvel_addr("left_leg_joint")] = np.pi / (2 * step_duration)
    vel_targets[i, model.get_joint_qvel_addr("right_leg_joint")] = -np.pi / (2 * step_duration)

# Define a simple PD controller to move the joints towards the desired targets
kp = 1.0
kd = 0.1
control_forces = np.zeros(model.nv)
for i in range(num_steps):
    # Get the current joint positions and velocities
    qpos = sim.data.qpos
    qvel = sim.data.qvel

    # Compute the control forces using a PD controller
    for j in range(model.nv):
        control_forces[j] = kp * (pos_targets[i, j] - qpos[j]) - kd * qvel[j]

    # Apply the control forces to the model
    sim.data.ctrl[:] = control_forces

    # Advance the simulation by a small time step
    sim.step()