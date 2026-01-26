# X2 Humanoid Robot Training Setup

## Overview
This document describes the training setup for the X2 humanoid robot, created based on the X1 configuration structure.

## File Structure Created

```
humanoid/envs/x2/
├── __init__.py                    # Package initialization
├── x2_dh_stand_config.py         # Configuration classes for X2
└── x2_dh_stand_env.py            # Environment implementation for X2
```

## Key Configurations

### Robot Specifications
- **Robot Name**: X2 humanoid
- **URDF File**: `x2_ultra_simple_collision.urdf` (as specified)
- **Controlled DOFs**: 15 (12 leg joints + 3 waist joints)
  - Left leg: 6 DOFs (hip pitch/roll/yaw, knee, ankle pitch/roll)
  - Right leg: 6 DOFs (hip pitch/roll/yaw, knee, ankle pitch/roll)
  - Waist: 3 DOFs (yaw, pitch, roll)
- **Passive Joints**: Arms (12 DOFs) and Head (2 DOFs) - not controlled during training
- **Base Height Target**: 0.85m (adjusted for X2 morphology vs X1's 0.61m)

### Environment Parameters
- **Observation Dimensions**:
  - Single observation: 52 features
  - Frame stack: 66 (history frames)
  - Total observation size: 3432 (66 × 52)
  
- **Action Dimensions**: 15 (one per controlled joint)

- **Privileged Observation Dimensions**: 78 features per frame
  - Includes full proprioceptive data with noise simulation

### Training Configuration
- **Task Name**: `x2_dh_stand`
- **Environment Instances**: 4096 parallel environments
- **Episode Length**: 24 seconds
- **Control Decimation**: 10 (policy runs at 50Hz while simulation runs at 1000Hz)

### Joint Control Parameters
- **Stiffness**: Proportional to joint type
  - Hip joints: 30-40 Nm/rad
  - Knee joints: 100 Nm/rad
  - Ankle joints: 35 Nm/rad
  - Waist joints: 50 Nm/rad

- **Damping**: Proportional to joint type
  - Hip/ankle/waist: 0.5-5 Nm·s/rad
  - Knee: 10 Nm·s/rad

### Action Scale
- `0.5` radians per action unit
- Target angle = action_scale × action + default_angle

### Reward Function Components
The X2 environment includes the following rewards (same as X1):
- Reference joint position tracking
- Feet distance maintenance
- Foot slip minimization
- Air time promotion (for dynamic walking)
- Contact force regularization
- Base height control
- Velocity tracking (linear and angular)
- Action smoothness
- Torque/velocity penalties
- Collision avoidance

### Domain Randomization
Includes the following randomization parameters:
- Friction randomization: [0.2, 1.3]
- Joint friction: [0.01, 1.15]
- Joint damping: [0.3, 1.5]
- Motor offset: [-0.035, 0.035] rad
- Base mass variation: [-3, 3] kg
- COM displacement: ±0.05m in each direction
- Gain randomization: ×[0.8, 1.2]
- Periodic random pushes with curriculum (0 to 0.25s duration)

### Gaits Supported
During training, the robot learns multiple gaits with curriculum:
1. **Walk omnidirectional** - Primary gait for forward/backward/lateral movement
2. **Stand** - Static balance
3. Gaits can be extended with: walk_sagittal, walk_lateral, rotate, etc.

### Terrain Configuration
- **Type**: Trimesh (can be switched to plane or heightfield)
- **Curriculum**: Disabled by default
- **Types**: Flat, rough flat, slopes, stairs, discrete, wave terrain
- **Proportions**: Primarily flat and rough-flat during initial training

## Training Usage

To train X2 with the default configuration:

```bash
python humanoid/scripts/train.py --task x2_dh_stand
```

### Optional Training Parameters
```bash
# Train with specific device
python humanoid/scripts/train.py --task x2_dh_stand --physics_engine physx --device cuda

# Custom environment parameters
python humanoid/scripts/train.py --task x2_dh_stand --num_envs 2048

# Resume training from checkpoint
python humanoid/scripts/train.py --task x2_dh_stand --resume --load_run -1 --checkpoint -1
```

## Key Differences from X1

| Parameter | X1 | X2 |
|-----------|-----|-----|
| File | x1.urdf | x2_ultra_simple_collision.urdf |
| Controlled DOFs | 12 | 15 |
| Observation Size | 47→48 | 52 |
| Base Height Target | 0.61m | 0.85m |
| Task Name | x1_dh_stand | x2_dh_stand |
| Foot Name | ankle_roll | ankle_roll |
| Knee Name | knee_pitch | knee |

## Training Hyperparameters

- **Learning Rate**: 1e-5
- **Entropy Coefficient**: 0.001
- **Gamma (discount)**: 0.994
- **Lambda (GAE)**: 0.9
- **Number of Learning Epochs**: 2
- **Mini-batches**: 4
- **Steps per Environment**: 24
- **Max Iterations**: 20,000

## Observation Structure (52 features)

1. **Command Input** (5): sin(phase), cos(phase), vel_x, vel_y, ang_vel_yaw
2. **Joint Positions** (15): Normalized positions of all controlled joints
3. **Joint Velocities** (15): Velocities of all controlled joints
4. **Previous Actions** (15): Last action commands
5. **Angular Velocity** (3): Base angular velocity (roll, pitch, yaw)
6. **Base Euler Angles** (3): Base orientation angles

## Privileged Observation Structure (78 features)

Includes all observations above plus:
- Command input (5)
- Joint position errors (15)
- Joint velocities (15)
- Current actions (15)
- Position differences from reference (15)
- Base linear velocity (3)
- Base angular velocity (3)
- Base euler angles (3)
- Push force (2)
- Push torque (3)
- Environment friction (1)
- Body mass (1)
- Stance mask (2)
- Contact mask (2)

## Notes

1. **Arms and Head**: Currently passive (zero torque). Can be activated by:
   - Modifying `num_actions` and observation dimensions
   - Adding arm control joints to `default_joint_angles`
   - Adjusting control stiffness/damping for arm joints

2. **Collision Detection**: Set to penalize collisions on the pelvis

3. **Contact Indices**: Feet indices are automatically detected from the URDF

4. **Terrain Curriculum**: Currently disabled but can be enabled for progressive difficulty

5. **Frequency**: Simulation at 1kHz, policy control at 50Hz (10:1 decimation)

## References

- X1 configuration: `humanoid/envs/x1/x1_dh_stand_config.py`
- X2 URDF: `resources/robots/X2/x2_ultra_simple_collision.urdf`
- Base robot class: `humanoid/envs/base/legged_robot.py`
- Task registry: `humanoid/utils/task_registry.py`
