"""
    This file is responsible to store common variables
    which are using by other functionalities.
"""
import os

# Define absolute path for repository 
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
src_dir = os.path.join(root_dir, "src")
storage_path = os.path.join(root_dir, "storage")
dataset_path = os.path.join(storage_path, "dataset")
dataset_config_file = os.path.join(dataset_path, "dataset.conf")
train = os.path.join(dataset_path, "train")
test = os.path.join(dataset_path, "test")

# Load 'LABELS' and 'INPUT_SIGNAL_TYPES' after preparing the dataset

# debug message
print("\n" + "Dataset is now located at: " + dataset_path)
