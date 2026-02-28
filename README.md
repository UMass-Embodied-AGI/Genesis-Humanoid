# ExtremControl: Low-Latency Humanoid Teleoperation with Direct Extremity Control

[[Website]](https://extremcontrol.github.io/) [[Video]](https://youtu.be/9Qb57bzvzO4) [[Paper](https://arxiv.org/abs/2602.11321)]

<p align="left">
  <img src="media/teaser.gif" width="480">
</p>

This is the branch for [ExtremControl](https://extremcontrol.github.io/). New feature of Genesis Humanoid will not be updated here.

## Installation & Example

**MacOS is supported.**

### 1. Install Dependencies

```bash
# Install uv (fast Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and setup the repository
git clone https://github.com/UMass-Embodied-AGI/Genesis-Humanoid.git
cd Genesis-Humanoid
git fetch origin extremcontrol:extremcontrol
git checkout extremcontrol
```

### 2. Install the `gs-env` package

```bash
# uv
uv sync --package gs-env

# pip
uv pip compile pyproject.toml -o requirements.txt --python /PATH/TO/PYTHON
pip install -r requirements.txt
```

### 3. Activate the environment

```bash
source .venv/bin/activate
```

### 4. Evaluate with provided checkpoint

```bash
python examples/run_ppo_motion.py --exp_name extremcontrol --eval True --show_viewer True
```

### 5. Setup real-world environment

Please refer to READMEs in each folder (TODO) under `src/env/gs_env/real`.

### 6. Deploy on real robot

```bash
uv pip install redis
redis-server
# Simulation
python deploy/g1_teleop.py --exp_name extremcontrol

# Real world
# Better to start from small ACTION_SCALE
python deploy/g1_teleop.py --exp_name extremcontrol --sim False --action_scale ACTION_SCALE
```

## Usage

### Process existing motions

```bash
# LAFAN1 (Optional but recommended)
python examples/convert_lafan.py

# HuB (Optional)
python examples/convert_hub.py

# Recorded MoCap motion
python examples/convert_optitrack.py

# AMASS (Optional)
# Download SMPLX body model to assets/body_models
# Download AMASS dataset to assets/AMASS
python examples/convert_amass.py
```

### Run RL training

```bash
# Teleop teacher policy
python examples/run_ppo_motion.py \
    --exp_name TEACHER_EXP_NAME \
    --env_name g1_motion_teacher \
    --env.motion_file assets/motion/motion.yaml
```

### Run BC distillation

```bash
# Distill a deployable policy
python examples/run_bc_motion.py \
    --exp_name BC_EXP_NAME \
    --env_name g1_motion \
    --teacher_exp_name TEACHER_EXP_NAME \
    --env.motion_file assets/motion/motion.yaml
```

### RL finetune / resume training

```bash
# Finetune the distilled policy
python examples/run_ppo_motion.py \
    --exp_name BC_EXP_NAME \
    --env_name g1_motion \
    --resume True \
    --use_stored_config False \
    --runner.freeze_actor_iterations 200 \
    --algo.lr 3e-5 \
    --env.motion_file assets/motion/teacher.yaml
```

### Evaluate trained policy

```bash
# Evaluation will store a deployable policy to deploy/logs/EXP_NAME
python examples/run_ppo_motion.py \
    --exp_name EXP_NAME \
    --num_ckpt NUM_CKPT (optional) \
    --eval True \
    --show_viewer True \
    --env.motion_file assets/motion/evaluate.pkl
```

### Deploy to a real robot

Install [unitree-sdk2-python](https://github.com/unitreerobotics/unitree_sdk2_python) for deployment.
Install [redis](https://github.com/redis/redis) for teleoperation.

```bash
# Make sure deploy/logs/EXP_NAME exists
# Sanity check in simulation
uv pip install redis
python deploy/g1_teleop.py --exp_name EXP_NAME
python deploy/g1_motion.py \
    --exp_name EXP_NAME \
    --motion_file MOTION_PATH

# Deployment should start from small ACTION_SCALE
python deploy/g1_teleop.py \
    --exp_name EXP_NAME \
    --sim False \
    --action_scale ACTION_SCALE
python deploy/g1_motion.py \
    --exp_name EXP_NAME \
    --motion_file MOTION_PATH \
    --sim False \
    --action_scale ACTION_SCALE
```

### Teleoperate the robot

```bash
# Test with existing motion
python deploy/motion_publisher.py --motion_file MOTION_FILE

# Optitrack MoCap
python deploy/optitrack_publisher,py

# SteanVR
python deploy/steamvr_publisher.py
```

## Citation

If you find our code useful, please consider citing our related paper:
```
@misc{xiong2026extremcontrol,
      title={ExtremControl: Low-Latency Humanoid Teleoperation with Direct Extremity Control}, 
      author={Ziyan Xiong and Lixing Fang and Junyun Huang and Kashu Yamazaki and Hao Zhang and Chuang Gan},
      year={2026},
      eprint={2602.11321},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2602.11321}, 
}
``` 

## Acknowledgement

The entire codebase is built on [GenesisPlayground](https://github.com/yun-long/GenesisPlayground). We thank [GMR](https://github.com/YanjieZe/GMR) for serving as a reference for retargeting and OptiTrack streaming. The human datasets used in this project includes [AMASS](https://amass.is.tue.mpg.de/), [HuB](https://hub-robot.github.io/) and [LAFAN1](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset).
