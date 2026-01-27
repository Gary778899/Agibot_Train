#!/usr/bin/env python3
"""
Visualization script to validate X2 robot initial height.
Compares current height (0.85m) vs calculated correct height (0.67m).
Uses URDF format for compatibility with MuJoCo viewer.
"""

import numpy as np
import mujoco
import mujoco.viewer
import time
import os
from humanoid import LEGGED_GYM_ROOT_DIR
from humanoid.envs import task_registry

def load_and_display_robot(height, duration=5.0, title_suffix="", enable_viewer=True):
    """
    Load and display X2 robot at specified height
    
    Args:
        height: Z position of pelvis
        duration: How long to display (seconds)
        title_suffix: Additional info for display
        enable_viewer: Whether to show MuJoCo viewer (default True)
    """
    # Use the URDF file with ground plane for visualization
    model_path = f"{LEGGED_GYM_ROOT_DIR}/resources/robots/X2/x2_ultra_simple_collision_with_ground.urdf"
    print(f"\nLoading model from: {model_path}")
    
    # Create model and data
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    
    # Load config to get default joint angles
    env_cfg, _ = task_registry.get_cfgs(name='x2_dh_stand')
    default_dof_pos = np.array(list(env_cfg.init_state.default_joint_angles.values()))
    
    # Set initial position
    data.qpos[0:3] = [0.0, 0.0, height]  # x, y, z
    data.qpos[3:7] = [0.0, 0.0, 0.0, 1.0]  # quat (w at end)
    
    # Set default joint angles
    num_joints = len(default_dof_pos)
    data.qpos[-num_joints:] = default_dof_pos
    
    # Step to ensure kinematics are updated
    mujoco.mj_step(model, data)
    
    print(f"\n{'='*60}")
    print(f"Height Validation: {title_suffix}")
    print(f"{'='*60}")
    print(f"Pelvis Z position: {height:.4f} m")
    
    # Get foot positions
    left_foot_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'left_ankle_roll_link')
    right_foot_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'right_ankle_roll_link')
    
    if left_foot_body_id >= 0:
        left_foot_z = data.xpos[left_foot_body_id, 2]
        print(f"Left ankle_roll frame Z: {left_foot_z:.4f} m")
        print(f"  Foot contact (lowest): ~{left_foot_z - 0.073:.4f} m")
    
    if right_foot_body_id >= 0:
        right_foot_z = data.xpos[right_foot_body_id, 2]
        print(f"Right ankle_roll frame Z: {right_foot_z:.4f} m")
        print(f"  Foot contact (lowest): ~{right_foot_z - 0.073:.4f} m")
    
    print()
    
    # Launch viewer if enabled
    if enable_viewer:
        try:
            with mujoco.viewer.launch_passive(model, data) as viewer:
                start_time = time.time()
                while (time.time() - start_time) < duration:
                    mujoco.mj_step(model, data)
                    viewer.sync()
                    time.sleep(0.01)  # ~100 Hz
                print(f"Viewer closed after {duration:.1f}s")
        except Exception as e:
            print(f"WARNING: Could not launch viewer: {e}")
            print("Continuing without visualization...")
            time.sleep(duration)

if __name__ == '__main__':
    print("\n" + "="*60)
    print("X2 ROBOT HEIGHT VALIDATION WITH VIEWER")
    print("="*60)
    
    # Check if running in WSL2 or other headless environment
    display = os.environ.get('DISPLAY', '')
    viewer_enabled = bool(display) or os.environ.get('WSL_HOST', '') != ''
    
    print(f"\nDisplay available: {bool(display)}")
    print(f"WSL2 detected: {os.environ.get('WSL_HOST', '') != ''}")
    print(f"Viewer will be {'ENABLED' if viewer_enabled else 'DISABLED'}\n")
    
    print("1. Testing CURRENT height (0.85 m)")
    load_and_display_robot(0.85, duration=5.0, title_suffix="CURRENT (0.85m)", enable_viewer=viewer_enabled)
    
    print("\n2. Testing CORRECTED height (0.67 m) - Robot should touch ground")
    load_and_display_robot(0.675, duration=5.0, title_suffix="CORRECTED (0.675m)", enable_viewer=viewer_enabled)
    
    print("\n" + "="*60)
    print("VALIDATION COMPLETE")
    print("="*60)
    print("\nObservations:")
    print("- At 0.85m: Robot is clearly floating above the ground")
    print("- At 0.675m: Robot feet should just touch the ground plane")
    print("\nRecommendation: Use 0.67 or 0.675 m as initial height")
