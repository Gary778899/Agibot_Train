# X1 Robot Training Pipeline - Comprehensive Learning Guide

## Table of Contents
1. **Fundamentals** - Core concepts
2. **Environment Layer** - How the robot environment is built
3. **Task Definition** - Observations, Actions, Rewards
4. **Algorithm Layer** - PPO and Neural Networks
5. **Training Pipeline** - The complete training flow
6. **Inference & Deployment** - Running trained models
7. **Customization** - Adapting to your robot

---

## SECTION 1: FUNDAMENTALS

### 1.1 Isaac Gym Basics

**What Isaac Gym Does:**
- Runs multiple physics simulations in parallel on GPU
- Each simulation is independent but runs in parallel
- Returns tensor data compatible with PyTorch

**Why It's Efficient:**
```
Traditional CPU Approach:
  Env 1 → Sim → Get obs, step → Wait
  Env 2 → Sim → Get obs, step → Wait
  ... (sequential, slow)

Isaac Gym GPU Approach:
  [Env 1, Env 2, Env 3, ..., Env 4096] → GPU Physics → All done in parallel!
```

**Result**: Train on 4096 environments simultaneously = much faster data collection

---

### 1.2 The RL Loop

Every RL agent does this repeatedly:

```
1. OBSERVE: Get state from environment
   └─ For X1: joint angles, velocities, IMU data, contact info, etc.
   
2. ACT: Policy network decides what to do
   └─ Policy: observation → action
   └─ For X1: 12 joint target angles
   
3. STEP: Apply action in simulator
   └─ Physics engine updates for 0.001 seconds
   
4. REWARD: Get feedback on how good the action was
   └─ Is it walking? (+reward)
   └─ Is it falling? (-reward)
   
5. LEARN: Update policy network to maximize rewards
   └─ PPO algorithm computes gradients
   └─ Network parameters updated via backprop
```

This repeats for thousands of iterations until the robot learns to walk.

---

### 1.3 Understanding the X1 Configuration

Looking at `x1_dh_stand_config.py`, here's what the task is:

```python
class env:
    num_envs = 4096                    # Run 4096 simulations in parallel
    num_observations = int(66 * 47)    # 3,102 total observations
    num_single_obs = 47                # 47 observations per timestep
    frame_stack = 66                   # Stack 66 timesteps of history
    
    num_actions = 12                   # Control 12 joints
    episode_length_s = 24              # Each episode runs 24 seconds
```

**What This Means:**
- The robot gets 47 sensor readings per timestep
- But the policy receives history: last 66 timesteps = 66 × 47 = 3,102 observations
- This history helps the policy understand motion patterns
- The policy outputs 12 numbers: target angles for the 12 joints

---

### 1.4 Key Parameters Explained

```python
class control:
    control_type = 'P'                 # Proportional control (PD controller)
    stiffness = {...}                  # How stiff the joints are
    damping = {...}                    # How damped (friction-like)
    action_scale = 0.5                 # Scale actions: -1 to +1 → -0.5 to +0.5 radians
    decimation = 10                    # Control at 50Hz (simulation runs at 1000Hz)

class sim:
    dt = 0.001                         # Simulation timestep: 1ms
    # So decimation=10 means: update control every 10ms (100Hz), sim at 1000Hz
```

**How Actions Work:**
```
Network output (raw): [-1.0, ..., +1.0]
                          ↓
Scale by action_scale: [-0.5, ..., +0.5]
                          ↓
Add to default_joint_angles: [default + scaled_action]
                          ↓
Send to PD controller: tracks target angle with stiffness/damping
                          ↓
Apply torques to joints in physics simulation
```

---

### 1.5 Rewards: Teaching the Robot to Walk

The X1 is taught using a **reward function**:

```python
class rewards:
    # Positive rewards (encourage these behaviors)
    tracking_lin_vel = 1.8             # Track forward speed commands
    tracking_ang_vel = 1.1             # Track rotation speed commands
    feet_air_time = 1.2                # Lift feet off ground (makes walking, not sliding)
    feet_distance = 0.2                # Space feet apart appropriately
    
    # Negative rewards (discourage these behaviors)
    action_smoothness = -0.002         # Penalize jerky movements
    torques = -8e-9                    # Penalize using too much torque
    dof_vel = -2e-8                    # Penalize high joint velocities
    collision = -1.0                   # Strong penalty for falling
    dof_pos_limits = -10.0             # Strong penalty for joint limit violations
```

**How Reward Works Each Step:**
```
reward = (1.8 * vel_tracking_error +
          1.1 * ang_tracking_error +
          1.2 * feet_air_time_reward +
          ... + other terms)

During training, the policy learns:
  - If I walk forward → +reward
  - If I stumble → -reward
  - If my actions are smooth → small +reward
```

---

## SECTION 2: ENVIRONMENT LAYER

### 2.1 Class Hierarchy

The code uses inheritance to build the environment:

```
BaseConfig (abstract)
    ↓
LeggedRobotCfg (base robot config)
    ↓
X1DHStandCfg (X1-specific config)
```

And for the actual environment:

```
VecEnv (abstract interface)
    ↓
LeggedRobot (base class, Isaac Gym integration)
    ↓
X1DHStandEnv (X1-specific implementation)
```

**Why This Design?**
- Reuse common code
- Easy to add new robots: just extend LeggedRobotCfg and LeggedRobot
- Abstract interface (VecEnv) allows any algorithm to work

---

### 2.2 The VecEnv Interface

[See vec_env.py]

This is the **contract** between environment and algorithm:

```python
class VecEnv(ABC):
    # These MUST be set by any implementing environment:
    num_envs: int                      # How many parallel envs
    num_obs: int                       # Observation vector size
    num_actions: int                   # Action vector size
    
    # These MUST be filled each step:
    obs_buf: torch.Tensor              # Current observations for all envs
    rew_buf: torch.Tensor              # Rewards for all envs
    reset_buf: torch.Tensor            # Which envs need reset
    episode_length_buf: torch.Tensor   # Current episode length for each env
    
    # These MUST be implemented:
    def step(self, actions):           # Apply actions, return (obs, reward, dones, extras)
    def reset(self, env_ids):          # Reset specific environments
    def get_observations(self):        # Return current observations
```

**Key Insight**: Any environment (Isaac Gym, MuJoCo, CoppeliaSim) can be wrapped to match this interface, then the same RL algorithm works!

---

## SECTION 3: TASK DEFINITION - Observations, Actions, Rewards

### 3.1 Constructing Observations

This is where the robot "sees":

```python
# Single observation at one timestep (47 values):
# [joint_angles(12), joint_velocities(12), base_linear_vel(3), 
#  base_angular_vel(3), base_quaternion(4), gravity_vector(3), 
#  foot_contacts(4), command(5)]
```

Let's look at what each means:

```
1. Joint Angles (12): θ1, θ2, ..., θ12
   └─ Current angle of each joint
   └─ Normalized by dividing by limits

2. Joint Velocities (12): θ̇1, θ̇2, ..., θ̇12
   └─ How fast each joint is moving
   └─ Scaled by 0.05 (see normalization)

3. Base Linear Velocity (3): vx, vy, vz
   └─ Robot's forward/side/vertical speed
   └─ From IMU or state estimation
   └─ Scaled by 2.0

4. Base Angular Velocity (3): ωx, ωy, ωz
   └─ Robot's rotation rates
   └─ Scaled by 1.0

5. Base Quaternion (4): qx, qy, qz, qw
   └─ Robot's orientation in 3D space
   └─ Normalized (always between -1 and 1)

6. Gravity Vector (3): gx, gy, gz
   └─ Direction of gravity in robot's frame
   └─ Helps robot know which way is "up"

7. Foot Contacts (4): binary for each foot
   └─ Is each foot touching ground?
   └─ Critical for determining walking phase

8. Commands (5): vx_cmd, vy_cmd, ωz_cmd, sin(yaw), cos(yaw)
   └─ What the human (or test) wants robot to do
   └─ Commands are sampled during training
   └─ Sine/cosine of yaw helps with circular commands
```

**Frame Stacking**: Why history?

```python
frame_stack = 66
num_observations = 66 * 47 = 3,102

This means the policy sees:
[obs_t-65, obs_t-64, ..., obs_t-1, obs_t]  ← Last 66 timesteps

Why?
- Single observation doesn't show motion
- History shows velocity, acceleration, patterns
- Network can learn: "If angle increasing and velocity increasing,
  probably falling → apply opposite torque"
```

---

### 3.2 Actions: What the Policy Controls

```python
num_actions = 12

Each action is in [-1, +1]:
  action = [-1.0, ..., +1.0]

Converted to joint target angles:
  target_angle[i] = default_angle[i] + action[i] * action_scale
  
With action_scale = 0.5:
  target_angle[i] = default_angle[i] + action[i] * 0.5
  
So action of -1 gives:    default - 0.5 radians
   action of  0 gives:    default
   action of +1 gives:    default + 0.5 radians
```

**The PD Controller** (Proportional-Derivative):

```
After network outputs action → target angle, what happens?

Isaac Gym applies a PD controller:

  torque = Kp * (target_angle - current_angle) + Kd * (0 - current_velocity)
  
Where:
  Kp = stiffness (e.g., 30 for hip_pitch_joint)
  Kd = damping (e.g., 3 for hip_pitch_joint)
  
Example:
  - Target: 0.5 rad, Current: 0.3 rad, Velocity: 0.1 rad/s
  - Error: 0.2 rad
  - torque = 30*0.2 + 3*(-0.1) = 6.0 - 0.3 = 5.7 Nm
```

This controller ensures smooth, stable tracking of network commands.

---

### 3.3 The Complete Reward Function

[See x1_dh_stand_config.py rewards section]

The reward combines many objectives:

```python
total_reward = (
    1.8 * lin_vel_tracking_reward +          # Primary: follow speed command
    1.1 * ang_vel_tracking_reward +          # Primary: follow rotation command
    1.2 * feet_air_time_reward +             # Make walking gait
    0.2 * feet_distance_reward +             # Keep feet apart
    0.3 * feet_rotation_reward +             # Proper foot orientation
    
    -0.1 * foot_slip_penalty +               # Don't slide feet
    -0.002 * action_smoothness_penalty +     # Smooth movements
    -8e-9 * torque_penalty +                 # Energy efficiency
    -2e-8 * dof_velocity_penalty +           # Don't jerk
    
    -1.0 * collision_penalty +               # STRONG: don't fall
    -10.0 * joint_limit_penalty              # STRONG: respect limits
)
```

**Training Insight**:
- Early training: robot uses all rewards equally
- As training progresses: robot learns collision penalty is strongest
- Final behavior: naturally stable, doesn't risk falling

---

## Key Takeaways from Sections 1-3

Before moving to algorithms, solidify these concepts:

1. **Parallelization**: 4096 envs run simultaneously on GPU
2. **Observations**: 3,102 values = 66 timesteps × 47 sensor readings
3. **Actions**: 12 network outputs → 12 joint target angles
4. **Control**: PD controller tracks target angles
5. **Rewards**: Sum of multiple objectives, robot learns to maximize total
6. **History Matters**: Network sees past 66 timesteps to understand motion

---

## SECTION 4: ALGORITHM LAYER - PPO & NEURAL NETWORKS

### 4.1 What is PPO (Proximal Policy Optimization)?

PPO is an RL algorithm that learns by:
1. **Collecting experiences**: Run policy in environment, save transitions
2. **Computing advantages**: How much better was this action than average?
3. **Updating policy**: Gradient descent to maximize expected rewards
4. **Staying close to old policy**: Don't change too much per iteration

**Why PPO?**
- More stable than earlier algorithms (like vanilla policy gradient)
- Sample efficient: can use same data multiple times
- Works well with continuous actions (like robot joints)

---

### 4.2 The Three Networks

The X1 uses THREE neural networks working together:

```
1. ACTOR (Policy Network)
   Input: 235 observations (stacked history)
   Output: 12 action means
   Architecture: [235] → [512] → [256] → [128] → [12]
   
   Purpose: Decide what the robot should do

2. CRITIC (Value Network)
   Input: 219 privileged observations
   Output: 1 value estimate
   Architecture: [219] → [768] → [256] → [128] → [1]
   
   Purpose: Estimate "how good is this state?"

3. STATE ESTIMATOR (Auxiliary Network)
   Input: 235 (last 5 timesteps of obs)
   Output: 3 (estimated linear velocity)
   Architecture: [235] → [256] → [128] → [64] → [3]
   
   Purpose: Learn to estimate true state from sensors
            (helps during sim-to-real transfer)
```

**Key Difference: Actor Input vs Critic Input**

```
Actor gets:
  - Full observation history (66 timesteps)
  - Last 5 timesteps separately
  - CNN-encoded long history
  - State estimator output
  = 235 total observations

Critic gets:
  - Privileged observations (ground truth state)
  - Not available during inference!
  = 219 values

Why? During training we have perfect state info (privileged obs),
so critic can estimate value accurately. During deployment, actor
must work with only its sensor observations.
```

---

### 4.3 The PPO Training Loop

Here's what happens every iteration (24 environments steps):

```
STEP 1: ROLLOUT (Collect Data)
───────────────────────────────
for i in range(24):  # 24 steps per env
    obs = current observations (235 values)
    action = actor(obs)  # Network predicts action
    obs_new, reward, done = env.step(action)  # Simulate
    
    Store: (obs, action, reward, obs_new, done)
    
Result: 4096 envs × 24 steps = 98,304 transitions

STEP 2: COMPUTE ADVANTAGES & RETURNS
─────────────────────────────────────
For each transition:
    V(s) = critic(obs)  # Value estimate "goodness of state"
    
    # Temporal Difference (TD) Error
    TD_error = reward + γ * V(s_next) - V(s)
    
    # Generalized Advantage Estimation (GAE)
    Advantage = sum of discounted TD errors with λ smoothing
               λ = 0.9 (smooths between TD and Monte Carlo)
    
    Return = V(s) + Advantage
    
Why advantages? 
  - Shows how much better/worse this action was
  - Reduces variance in policy gradient estimates
  - λ = 0.9 balances bias and variance

STEP 3: UPDATE NETWORKS (Multiple mini-batches)
────────────────────────────────────────────────
Split 98,304 transitions into 4 mini-batches
For each mini-batch, perform:

    # Compute new policy and value predictions
    new_action_log_prob = actor(obs_batch)
    new_value = critic(critic_obs_batch)
    
    # PPO Clipped Surrogate Loss
    ratio = exp(new_log_prob - old_log_prob)
    
    L_clip = -min(
        ratio * advantages,
        clamp(ratio, 1-ε, 1+ε) * advantages
    )
    where ε = 0.2 (clip parameter)
    
    Purpose: If ratio > 1, network wants to take more of this action.
            Clamp prevents too large a change (stays close to old policy).
    
    # Value Function Loss
    L_value = (new_value - return)²
    
    # Entropy Loss (Exploration bonus)
    L_entropy = -entropy_coef * entropy(distribution)
    
    # State Estimator Loss
    L_state = MSE(estimated_vel, privileged_vel)
    
    # Total Loss
    L_total = L_clip + 0.5 * L_value + 0.001 * L_entropy + L_state
    
    # Gradient Update
    θ ← θ - α * ∇L_total  (α = learning_rate = 1e-5)

This is done 2 times (num_learning_epochs = 2)
```

---

### 4.4 Understanding PPO Clipping

This is the most important part of PPO:

```
Old policy:     P_old(action|obs) = probability of action under old network
New policy:     P_new(action|obs) = probability of action under new network
Ratio:          r = P_new / P_old

Good action (positive advantage):
  r = 1.0  → ratio=1, keep probability same ✓
  r = 1.5  → ratio=1.5, increase probability by 50% 
           → but clip to 1.2, increase by 20% max ✓ (trust region)
  r = 0.5  → ratio=0.5, cut probability by 50% → keep clipped to 0.8 ✓

Bad action (negative advantage):
  r = 1.5  → ratio=1.5, but clip to 0.8, reduce by 20% min ✓
  r = 0.3  → ratio=0.3, reduce probability by 70% 
           → keep clipped to 0.8, reduce by 20% only ✓
```

**Result**: Network can't change policy too much in one iteration
→ More stable training
→ No catastrophic policy collapse

---

### 4.5 Privileged Information & Distillation

This is unique to this framework:

```
TRAINING:
┌──────────────────┐
│ Privileged Obs   │  (Ground truth state from simulator)
│ (219 values)     │
└──────┬───────────┘
       │
       ↓
  ┌─────────────┐
  │   CRITIC    │  ← Learns optimal value function
  │  (768-256)  │
  └─────────────┘

┌──────────────────────────────┐
│ Regular Obs                  │  (What sensors provide)
│ (235 values, includes        │
│  sensor noise + domain rand) │
└──────┬───────────────────────┘
       │
       ├──→ CNN → [64]
       │
       ├──→ STATE ESTIMATOR → [3 linear vel]
       │
       ├──→ Concatenate
       │
       ↓
  ┌──────────────────────┐
  │     ACTOR            │  ← Learns from noisy observations
  │    (512-256-128)     │
  └──────────────────────┘
```

**Key Insight**: 
- State estimator learns to estimate **linear velocity** from sensors
- During training, it's supervised by privileged obs
- During deployment, it estimates velocity from sensors alone
- Actor learns to trust state estimator's estimates

This is **sim-to-real transfer**: trained policy works on real robot 
because it learned to estimate state from noisy sensors.

---

### 4.6 The Complete Training Step (Code Flow)

```python
# Simplified pseudocode of one training iteration

# 1. ROLLOUT
for step in range(24):
    obs = env.get_observations()           # 4096 envs × 235
    critic_obs = env.get_privileged_obs()  # 4096 envs × 219
    
    with torch.inference_mode():  # No gradients
        action = actor(obs)  # 4096 × 12
        value = critic(critic_obs)  # 4096 × 1
        action_log_prob = distribution.log_prob(action)  # 4096
    
    obs_new, rewards, dones = env.step(actions)  # Physics step
    storage.add(obs, action, reward, value, action_log_prob)

# 2. COMPUTE ADVANTAGES
last_value = critic(obs_new)
storage.compute_returns(last_value, gamma=0.994, lambda=0.9)

# 3. UPDATE (Mini-batch training)
for epoch in range(2):
    for batch in mini_batch_generator(4):  # 4 mini-batches
        
        # Forward pass
        new_actions = actor(obs_batch)
        new_values = critic(critic_obs_batch)
        new_log_probs = get_log_prob(new_actions)
        
        # Loss computation
        ratio = exp(new_log_probs - old_log_probs)
        L_clip = -min(ratio * adv, clamp(ratio, 0.8, 1.2) * adv)
        L_value = (new_values - returns)²
        
        L_total = L_clip + 0.5 * L_value
        
        # Backward pass
        L_total.backward()
        clip_grad_norm(0.5)
        optimizer.step()

# Result: Network slightly improved, ready for next iteration
```

---

## SECTION 5: TRAINING PIPELINE - COMPLETE FLOW

### 5.1 The Entry Point: train.py

```python
# scripts/train.py is extremely simple:

from humanoid.envs import *
from humanoid.utils import get_args, task_registry

def train(args):
    # Create environment and get config
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    
    # Create algorithm + runner
    ppo_runner, train_cfg, log_dir = task_registry.make_alg_runner(
        env=env, name=args.task, args=args
    )
    
    # Train!
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations)
```

**What Happens:**
```
task_registry.make_env()
  ↓
  Returns: X1DHStandEnv initialized with X1DHStandCfg
  ↓
  4096 parallel environments ready in Isaac Gym

task_registry.make_alg_runner()
  ↓
  Creates ActorCriticDH networks
  Creates DHPPO algorithm
  Creates DHOnPolicyRunner training loop
  ↓
  Ready to run training

ppo_runner.learn()
  ↓
  Main training loop (explained below)
```

---

### 5.2 The Training Loop (DHOnPolicyRunner.learn)

This is the heart of the whole system:

```python
# Pseudocode of main training loop

for iteration in range(num_iterations):  # Up to 20,000 iterations
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PHASE 1: ROLLOUT (Collect experiences)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    for step in range(24):  # 24 steps per iteration
        
        # Get current state
        obs = env.get_observations()              # [4096, 235]
        critic_obs = env.get_privileged_obs()     # [4096, 219]
        
        # Policy step
        with torch.inference_mode():  # No gradient tracking
            
            # Actor network predicts action
            action = actor(obs)                   # [4096, 12]
            
            # Critic estimates value of current state
            value = critic(critic_obs)            # [4096, 1]
            
            # Get probability of action
            action_log_prob = policy.log_prob(action)  # [4096]
        
        # Physics step
        obs_next, reward, done = env.step(action)
        
        # Store transition for later learning
        storage.add(
            obs=obs,
            action=action, 
            reward=reward,
            value=value,
            log_prob=action_log_prob,
            done=done
        )
        
        # Track episode stats
        if episode_done:
            log_episode_reward()
    
    # At end of rollout: 4096 envs × 24 steps = 98,304 transitions
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PHASE 2: COMPUTE ADVANTAGES AND RETURNS
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    # Bootstrap: what's the value of final state?
    last_value = critic(obs_final)  # [4096, 1]
    
    # For each transition, compute advantage
    # A(s,a) = r + γ*V(s') - V(s)
    #        + γ*λ*A(s',a') + ...  (λ = 0.9)
    storage.compute_returns(
        last_value,
        gamma=0.994,        # Discount future rewards
        lambda=0.9          # GAE parameter
    )
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PHASE 3: POLICY UPDATE (Mini-batch training)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    # Split 98,304 transitions into 4 mini-batches of ~24,576 each
    for epoch in range(2):  # Update 2 times on same data
        
        for batch in mini_batch_generator(4):  # 4 mini-batches
            
            obs_batch = batch['observations']
            critic_obs_batch = batch['critic_obs']
            action_batch = batch['actions']
            old_log_prob_batch = batch['log_probs']
            advantage_batch = batch['advantages']
            return_batch = batch['returns']
            
            # Forward pass with new policy
            new_action_mean = actor(obs_batch)          # [24576, 12]
            new_log_prob = distribution.log_prob(new_action_mean)  # [24576]
            new_value = critic(critic_obs_batch)        # [24576, 1]
            
            # ========== Compute Losses ==========
            
            # 1. PPO Clipped Surrogate Loss
            ratio = exp(new_log_prob - old_log_prob_batch)
            surrogate = -advantage_batch * ratio
            surrogate_clipped = -advantage_batch * clamp(
                ratio, 1-0.2, 1+0.2
            )
            loss_surrogate = max(surrogate, surrogate_clipped).mean()
            
            # 2. Value Function Loss
            loss_value = (new_value - return_batch)²
            if use_clipped_value_loss:
                value_clipped = target_value + (new_value - target_value).clamp(
                    -0.2, 0.2
                )
                loss_value = max(
                    (new_value - return_batch)²,
                    (value_clipped - return_batch)²
                ).mean()
            else:
                loss_value = (new_value - return_batch)².mean()
            
            # 3. State Estimator Loss (auxiliary)
            # Estimate linear velocity from short obs
            estimated_velocity = state_estimator(obs_batch[:, -235:])
            true_velocity = critic_obs_batch[:, 53:56]  # From privileged obs
            loss_state_est = MSE(estimated_velocity, true_velocity)
            
            # 4. Entropy Loss (exploration bonus)
            loss_entropy = -entropy(distribution).mean() * 0.001
            
            # ========== Total Loss ==========
            loss_total = (
                loss_surrogate +
                0.5 * loss_value +
                loss_entropy +
                loss_state_est
            )
            
            # ========== Backward Pass ==========
            optimizer.zero_grad()
            loss_total.backward()
            
            # Gradient clipping: prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(
                actor_critic.parameters(),
                max_norm=0.5
            )
            
            optimizer.step()
            
            # Track losses
            log_value_loss += loss_value.item()
            log_surrogate_loss += loss_surrogate.item()
            log_state_loss += loss_state_est.item()
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PHASE 4: LOGGING AND SAVING
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    # Log to tensorboard
    writer.add_scalar('Loss/value', mean_value_loss, iteration)
    writer.add_scalar('Loss/surrogate', mean_surrogate_loss, iteration)
    writer.add_scalar('Loss/state_estimator', mean_state_loss, iteration)
    writer.add_scalar('Performance/fps', total_fps, iteration)
    if episode_rewards:
        writer.add_scalar('Reward/mean', mean(episode_rewards), iteration)
    
    # Print progress
    print(f"Iter {iteration}: reward={mean_reward:.2f}, fps={fps:.0f}")
    
    # Save checkpoint every 100 iterations
    if iteration % 100 == 0:
        torch.save({
            'model': actor_critic.state_dict(),
            'optimizer': optimizer.state_dict(),
            'iteration': iteration
        }, f'logs/model_{iteration}.pt')
```

**Key Numbers:**
- 4096 environments × 24 steps = **98,304 transitions per iteration**
- 2 epochs × 4 mini-batches = **8 gradient updates** per iteration
- Each gradient update uses ~24,576 transitions
- Total: **~2.4M transitions** trained per iteration
- Training 20,000 iterations = **48 billion** transitions total
- Speed: ~10,000-15,000 **steps/second** on GPU

---

### 5.3 Why This Architecture Works Well

```
┌─────────────────────────────────────────────────────────────┐
│  ADVANTAGES OF THIS DESIGN                                  │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. SAMPLE EFFICIENT                                         │
│     - Use same 98k transitions 8 times (mini-batches)       │
│     - Less environment steps needed                          │
│                                                              │
│  2. STABLE LEARNING                                          │
│     - PPO clipping prevents catastrophic changes            │
│     - Advantage estimation reduces variance                 │
│     - Multiple epochs smooth out noise                       │
│                                                              │
│  3. SIM-TO-REAL TRANSFER                                     │
│     - State estimator learns from noisy sensors             │
│     - Privileged obs during training improves value fn      │
│     - Actor learns from realistic observations              │
│                                                              │
│  4. GPU ACCELERATION                                         │
│     - All 4096 envs on GPU simultaneously                   │
│     - Mini-batch training fully parallelized                │
│     - 10,000+ steps/second performance                      │
│                                                              │
│  5. DOMAIN RANDOMIZATION                                     │
│     - Friction, mass, com, motor delay all randomized       │
│     - Robot learns robust policy for real world             │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## SECTION 6: INFERENCE & DEPLOYMENT

### 6.1 Running Trained Policy (play.py)

```python
# scripts/play.py shows how to use trained policy

# 1. Load trained model
ppo_runner.load('logs/model_10000.pt')
policy = ppo_runner.get_inference_policy()

# 2. Run in environment
for step in range(max_steps):
    obs = env.get_observations()
    
    # Forward pass through policy
    action = policy(obs)  # No noise! Just mean action
    
    obs_next, reward, done = env.step(action)

# Key difference from training:
# - Training: action = mean + std * noise  (explores)
# - Inference: action = mean  (deterministic, reproduces)
```

### 6.2 Exporting to JIT (C++ compatible)

```python
# For real robot deployment, convert to TorchScript

policy_jit = torch.jit.script(actor_critic.act_inference)
torch.jit.save(policy_jit, 'policy.pt')

# Can now load and run from C++:
// C++ code
torch::jit::script::Module module = torch::jit::load("policy.pt");
auto output = module.forward({obs_tensor});
```

---

## SECTION 7: CUSTOMIZATION GUIDE - YOUR OWN ROBOT

Now that you understand the pipeline, here's how to adapt it to your robot:

### 7.1 Step-by-Step Adaptation

```
STEP 1: Prepare Robot Model
  └─ Create URDF file with correct joints, masses, frames
  └─ Create MJCF file for Mujoco sim2sim validation
  └─ Place in resources/robots/your_robot/

STEP 2: Create Config
  └─ Create humanoid/envs/your_robot/your_robot_config.py
  └─ Inherit from LeggedRobotCfg
  └─ Set:
    * num_actions = number of joints to control
    * num_observations = your state size
    * Control parameters (stiffness, damping, action_scale)
    * Reward weights (what behavior you want)
    * Observation scaling (normalize observations)

STEP 3: Create Environment
  └─ Create humanoid/envs/your_robot/your_robot_env.py
  └─ Inherit from LeggedRobot
  └─ Implement:
    * compute_observations() - what sensors provide
    * compute_privileged_observations() - ground truth (training only)
    * compute_reward() - define task rewards
    * reset_env() - reset to initial state

STEP 4: Register Task
  └─ Add to humanoid/envs/__init__.py
  └─ Example:
    from .your_robot.your_robot_env import YourRobotEnv
    from .your_robot.your_robot_config import YourRobotCfg, YourRobotCfgPPO

STEP 5: Train
  └─ python scripts/train.py --task=your_robot --run_name=test

STEP 6: Validate & Deploy
  └─ python scripts/play.py --task=your_robot --load_run=<date>test
  └─ python scripts/sim2sim.py --task=your_robot --load_model logs/.../
  └─ Deploy on real robot!
```

---

### 7.2 Observations Design (Critical!)

The success of training depends heavily on good observations.

```python
# Example: What to observe for humanoid locomotion

def compute_observations(self):
    obs_buf = torch.zeros(
        (self.num_envs, 47),  # 47 features per timestep
        device=self.device
    )
    
    idx = 0
    
    # 1. Joint States (24)
    obs_buf[:, idx:idx+12] = (self.joint_pos - self.default_pos) / joint_ranges
    idx += 12
    obs_buf[:, idx:idx+12] = self.joint_vel * 0.05  # Scale velocities
    idx += 12
    
    # 2. Base Motion (6)
    obs_buf[:, idx:idx+3] = self.base_lin_vel * 2.0  # Linear velocity
    idx += 3
    obs_buf[:, idx:idx+3] = self.base_ang_vel * 1.0  # Angular velocity
    idx += 3
    
    # 3. Orientation (4)
    obs_buf[:, idx:idx+4] = self.base_quat  # Quaternion
    idx += 4
    
    # 4. Gravity Vector (3)
    obs_buf[:, idx:idx+3] = gravity_vector_in_body_frame
    idx += 3
    
    # 5. Contact Info (4)
    obs_buf[:, idx:idx+4] = self.foot_contact_forces > contact_threshold
    idx += 4
    
    # 6. Commands (5)
    obs_buf[:, idx:idx+5] = self.commands[:, :5]
    
    return obs_buf
```

**Design Principles:**
1. **Include proprioception** (joint angles, velocities) - robot knows its own state
2. **Include exteroception** (base velocity, orientation) - robot knows external state
3. **Include contact info** - critical for walking phase
4. **Include commands** - what task is robot trying to do
5. **Normalize observations** - divide by typical ranges (see normalization section)
6. **Frame stack** - include history (last N timesteps)

---

### 7.3 Reward Function Design

This is where you teach the robot what you want!

```python
def compute_reward(self):
    reward = torch.zeros(
        self.num_envs,
        device=self.device
    )
    
    # ===== TASK REWARDS (What behavior you want) =====
    
    # 1. Velocity Tracking (Primary objective)
    lin_vel_error = torch.norm(
        self.base_lin_vel[:, :2] - self.commands[:, :2],
        dim=1
    )
    reward += 1.8 * torch.exp(-5 * lin_vel_error²)
    
    # 2. Rotation Tracking
    ang_vel_error = torch.abs(
        self.base_ang_vel[:, 2] - self.commands[:, 2]
    )
    reward += 1.1 * torch.exp(-5 * ang_vel_error²)
    
    # ===== GAIT REWARDS (Walking, not sliding) =====
    
    # Foot air time (makes walking gait, not sliding)
    reward += 1.2 * foot_air_time
    
    # Feet distance (keep feet apart)
    reward += 0.2 * feet_distance_reward
    
    # ===== STABILITY REWARDS =====
    
    # Base height (keep body upright)
    height_error = torch.abs(self.base_pos[:, 2] - target_height)
    reward += 0.2 * torch.exp(-5 * height_error²)
    
    # Orientation (keep level)
    roll_pitch_error = torch.norm(self.base_rpy[:, :2], dim=1)
    reward += 0.3 * torch.exp(-5 * roll_pitch_error²)
    
    # ===== ENERGY/EFFICIENCY PENALTIES =====
    
    reward -= 0.002 * action_smoothness  # Prefer smooth movements
    reward -= 8e-9 * torch.sum(torques²)  # Minimize torque
    reward -= 2e-8 * torch.sum(joint_vel²)  # Avoid jerky movements
    
    # ===== SAFETY PENALTIES (Strong!) =====
    
    # Contact penalty (don't touch anything except feet)
    for contact_body in bad_contact_bodies:
        contact_forces = get_contact(contact_body)
        reward[contact_forces > 0] -= 1.0
    
    # Joint limits (don't exceed joint ranges)
    at_limits = torch.abs(self.joint_pos) > limit
    reward[at_limits] -= 10.0
    
    return reward
```

**Key Principles:**
1. **Reward shaping**: Multiple objectives, scaled by importance
2. **Exponential rewards**: Smooth, continuous learning signal
3. **Penalties > Rewards**: Avoid bad behaviors more than reward good ones
4. **Curriculum**: Start easy (low commands), gradually increase difficulty
5. **Experiment**: Test different scales (1.8, 1.1, 0.2, etc.)

---

### 7.4 Domain Randomization (For Real World)

Make training robust to real-world variations:

```python
class domain_rand:
    # ===== PHYSICAL PARAMETERS =====
    randomize_friction = True
    friction_range = [0.2, 1.3]  # Vary ground friction
    
    randomize_base_mass = True
    added_mass_range = [-3, 3]  # Vary robot weight
    
    randomize_com = True
    com_displacement_range = [[-0.05, 0.05], [-0.05, 0.05], [-0.05, 0.05]]
    
    randomize_gains = True
    stiffness_multiplier_range = [0.8, 1.2]  # Vary joint stiffness
    damping_multiplier_range = [0.8, 1.2]    # Vary damping
    
    # ===== MOTOR DELAYS & LAGS =====
    add_lag = True  # Latency in motor control
    lag_timesteps_range = [5, 40]  # 5-40ms delay
    
    add_dof_lag = True  # Joint velocity delays
    dof_lag_timesteps_range = [0, 40]
    
    # ===== NOISE INJECTION =====
    # (See noise section in config)
    
    # ===== EXTERNAL DISTURBANCES =====
    push_robots = True  # Random pushes during training
    push_duration = [0.05, 0.1, 0.15, 0.2, 0.25]  # Increase over time
```

**Why This Matters:**
- Train on [friction 0.2 to 1.3] → robot works on different floors
- Train with mass ±3kg → works if you add payload
- Train with motor delays → smoother, more realistic behavior
- Train with random pushes → robust balance

---

### 7.5 Quick Checklist for Your Robot

```
□ URDF & MJCF files created
□ Config file created
  □ Set num_joints correctly
  □ Set default_joint_angles
  □ Set control parameters
  □ Design observations (what to observe)
  □ Design reward function
  □ Set domain randomization parameters

□ Environment file created
  □ Implement compute_observations()
  □ Implement compute_privileged_observations()
  □ Implement compute_reward()
  □ Implement reset_env()

□ Registered in __init__.py
□ Basic test: python scripts/train.py --task=your_robot --headless
□ Debugging:
  □ Check observations are reasonable scale (see obs_scales)
  □ Check rewards are not all zeros
  □ Check policy can move joints (check action_scale)
  □ Watch training: use play.py to visualize

□ Training
  □ Start with high learning rate, decrease if unstable
  □ Monitor loss curves in tensorboard
  □ Train for 5,000-20,000 iterations
  □ Validate behavior in play.py

□ Deployment
  □ Export to JIT
  □ Run sim2sim validation with Mujoco
  □ Deploy on real robot
```

---

## KEY TAKEAWAYS

**The Complete Picture:**

```
1. ENVIRONMENT (Isaac Gym)
   - 4096 parallel simulations
   - Physics timestep: 1ms (1000 Hz)
   - Control frequency: 50Hz (decimation=10)
   
2. OBSERVATIONS
   - 47 values per timestep
   - Stacked 66 timesteps = 3102 dimensions
   - Contains: joint states, base motion, orientation, contacts, commands
   
3. POLICY (Actor Network)
   - Input: 235 current observations (5 timesteps)
   - CNN processes long history (66 timesteps)
   - Outputs: 12 action means + log-standard deviations
   - Distribution: Gaussian for continuous actions
   
4. VALUE FUNCTION (Critic Network)
   - Input: 219 privileged observations (ground truth)
   - Outputs: Single value estimate
   - Used during training only (advantage estimation)
   
5. PPO ALGORITHM
   - Collect 98,304 transitions per iteration
   - Compute advantages using GAE (λ=0.9)
   - Mini-batch training (4 batches, 2 epochs)
   - Clipped surrogate loss for stability
   
6. REWARD FUNCTION
   - Velocity tracking (primary: 1.8)
   - Walking gait (feet air time: 1.2)
   - Energy efficiency (torques: -8e-9)
   - Safety (collision: -1.0, limits: -10.0)
   - Sum of weighted objectives
   
7. TRAINING
   - 20,000 iterations
   - ~2.4M transitions per iteration
   - ~48 billion transitions total
   - Saves checkpoint every 100 iterations
   - Typical training time: 2-6 hours on NVIDIA A100
   
8. DEPLOYMENT
   - Export to TorchScript JIT
   - Run on real robot at 50Hz
   - State estimator provides velocity estimate
   - Deterministic (no noise) during inference
```

---

## WHERE TO GO FROM HERE

1. **Study the Code:**
   - Read [legged_robot.py](humanoid/envs/base/legged_robot.py) - environment implementation
   - Read [x1_dh_stand_env.py](humanoid/envs/x1/x1_dh_stand_env.py) - task-specific details
   - Read [dh_ppo.py](humanoid/algo/ppo/dh_ppo.py) - algorithm details

2. **Run Training:**
   - `python scripts/train.py --task=x1_dh_stand --run_name=test --headless`
   - Monitor in tensorboard: `tensorboard --logdir logs/`

3. **Visualize Results:**
   - `python scripts/play.py --task=x1_dh_stand --load_run=<date>test`

4. **Adapt to Your Robot:**
   - Create your_robot_config.py
   - Create your_robot_env.py
   - Follow Section 7 checklist

5. **Advanced Topics to Explore:**
   - Curriculum learning (progressively harder tasks)
   - Domain randomization tuning
   - Reward engineering (shaping for better behaviors)
   - Multi-task learning (single policy for multiple gaits)
   - Real-to-sim improvement loops



