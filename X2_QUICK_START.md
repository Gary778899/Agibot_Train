# X2 Training Quick Start Guide

## What Was Created

Successfully created complete training infrastructure for X2 humanoid robot:

### Files Created
1. **[humanoid/envs/x2/x2_dh_stand_config.py](humanoid/envs/x2/x2_dh_stand_config.py)**
   - Configuration for X2 training
   - 15 DOF control (12 leg + 3 waist)
   - 4096 parallel environments
   - Includes reward specifications and domain randomization

2. **[humanoid/envs/x2/x2_dh_stand_env.py](humanoid/envs/x2/x2_dh_stand_env.py)**
   - Environment implementation for X2
   - All training dynamics and reward calculations
   - Gait planning and command sampling
   - 20+ reward functions

3. **[humanoid/envs/x2/__init__.py](humanoid/envs/x2/__init__.py)**
   - Package initialization file

### Files Modified
- **[humanoid/envs/__init__.py](humanoid/envs/__init__.py)** 
  - Added X2 imports
  - Registered X2 task with task_registry

## Quick Start

### Train X2
```bash
cd /home/fsich/project/Agibot_Train
python humanoid/scripts/train.py --task x2_dh_stand
```

### Resume Training
```bash
python humanoid/scripts/train.py --task x2_dh_stand --resume --load_run -1 --checkpoint -1
```

### Train with Custom Settings
```bash
# Smaller batch
python humanoid/scripts/train.py --task x2_dh_stand --num_envs 2048

# GPU device
python humanoid/scripts/train.py --task x2_dh_stand --device cuda:0
```

## Key Parameters

| Parameter | Value |
|-----------|-------|
| **URDF** | x2_ultra_simple_collision.urdf |
| **Controlled DOFs** | 15 (legs + waist) |
| **Actions** | 15 (one per DOF) |
| **Observations** | 52 features per frame × 66 frame history |
| **Environments** | 4096 parallel |
| **Control Freq** | 50 Hz |
| **Simulation Freq** | 1000 Hz |
| **Episode Length** | 24 seconds |
| **Max Iterations** | 20,000 |

## Robot Configuration

### Leg Control (12 DOFs)
- **Left Leg**: hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll
- **Right Leg**: hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll

### Waist Control (3 DOFs)
- **Waist**: yaw_joint, pitch_joint, roll_joint

### Passive (Not Controlled)
- **Arms** (12 DOFs): Shoulder/elbow/wrist per arm
- **Head** (2 DOFs): Yaw and pitch

## Observation Format (52 features)

```
[sin(phase), cos(phase),           # 2 - gait phase
 vx, vy, vz_ang,                    # 3 - velocity commands
 q[0-14],                           # 15 - joint positions (normalized)
 dq[0-14],                          # 15 - joint velocities
 last_action[0-14],                 # 15 - previous actions
 base_omega,                        # 3 - base angular velocity
 base_euler]                        # 3 - base orientation
```

## Training Improvements Made for X2

1. ✅ **Updated DOF Count**: 12 → 15 for leg + waist control
2. ✅ **Adjusted Observations**: 47 → 52 features to accommodate waist
3. ✅ **Morphology Adjustment**: Base height 0.61m → 0.85m
4. ✅ **Joint Defaults**: Added waist joint angles (all 0.0)
5. ✅ **Control Parameters**: Added stiffness/damping for waist joints
6. ✅ **Domain Randomization**: Extended for 15 joints
7. ✅ **Reference Trajectory**: Waist set to maintain neutral position

## Gaits Supported

The X2 can learn:
- **Omnidirectional walking** (forward/backward/lateral)
- **Standing balance** (static)
- **Rotational movements** (yaw)
- Can be extended with: sagittal walk, lateral walk

## Expected Training Performance

Initial training should show:
- **Episode 1-100**: Base stability and standing
- **Episode 100-500**: Walking backward/forward
- **Episode 500-2000**: Omnidirectional movement and smoothing
- **Episode 2000+**: Advanced gaits and robustness

Training time ~2-4 hours on high-end GPU for 20k iterations with 4096 envs.

## Monitoring Training

Check logs in: `logs/x2_dh_stand/` directory

Monitor key metrics:
- `rew_tracking_lin_vel` - Velocity tracking reward
- `rew_tracking_ang_vel` - Rotation tracking reward  
- `rew_feet_air_time` - Gait quality
- `rew_action_smoothness` - Motion smoothness
- `rew_base_height` - Posture stability

## Exporting Trained Policy

Once training is complete:
```bash
python humanoid/scripts/export_policy_dh.py --task x2_dh_stand --num_envs 1 --headless
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Out of memory | Reduce `num_envs` (e.g., 2048) |
| Robot falling | Check default joint angles |
| Slow training | Increase `num_envs` if GPU memory allows |
| Import errors | Verify all files created in x2 folder |

## Additional Resources

- Training script: [humanoid/scripts/train.py](humanoid/scripts/train.py)
- Detailed config: [X2_TRAINING_SETUP.md](X2_TRAINING_SETUP.md)
- X1 reference: [humanoid/envs/x1/](humanoid/envs/x1/)
- Base robot class: [humanoid/envs/base/legged_robot.py](humanoid/envs/base/legged_robot.py)
