import mujoco_py
import numpy as np

# Load the model from an XML file
model = mujoco_py.load_model_from_path("humanoid.xml")

# Create a simulation environment for the model
sim = mujoco_py.MjSim(model)

# Set the initial state of the model
sim.data.qpos[:] = np.zeros(model.nq)
sim.data.qpos[2] = 1.0  # Set the pelvis height to 1 meter
sim.data.qvel[:] = np.zeros(model.nv)
sim.data.ctrl[:] = np.zeros(model.nu)

# Step the simulation forward in time
sim.step()

# Visualize the model
sim.render()