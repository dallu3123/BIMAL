import argparse
import os

import cv2
import h5py
import IPython
import matplotlib.pyplot as plt
import numpy as np

e = IPython.embed

JOINT_NAMES = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]
STATE_NAMES = JOINT_NAMES + ["gripper"]

def load_hdf5(dataset_dir, dataset_name):
    dataset_path = os.path.join(dataset_dir, dataset_name + '.hdf5')

    if not os.path.exists(dataset_path):
        print(f"Dataset does not exist: \n{dataset_path}\n")
        exit()

    with h5py.File(dataset_path, "r") as root:
        is_sim = root.attrs["sim"]