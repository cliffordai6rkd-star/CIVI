<div align="center">
  <img src="images/LIEBRO-FT_logo2.png" width="70%">

  # LIBERO-FT: Evaluating Robotic Manipulation under Force-Related Domain Shifts
</div>


<div align="center">
  <img src="./images/k4_examples.gif" width="100%" alt="replay_k1_open_put">
</div>

## 📖 Overview

`LIBERO-FT` is a toolkit (including benchmark) developed for evaluating robotic manipulation under **force-related** domain shifts. 
Highly based on the LIBERO benchmark, it can be considered an extension that builds upon its core functionality. 
Key features include extracting 6D wrench force-torque data from the simulator (Mujoco), replaying and saving benchmark trajectories, 
and modifying physical properties like gripper stiffness and friction to simulate domain shifts (based on [Robosuite](https://robosuite.ai/)). 
This repository aims to provide a framework for testing and evaluating the adaptability of robotic models to 
varying physical conditions, enabling thorough analysis before deploying force-based policies into real-world scenarios.


## ✨ Features

- **Read Mujoco 6D force-torque data (MujocoSensorReader)**: Reads raw 6D wrench force-torque data from the native Mujoco sensors and exposes it to the LIBERO `OffScreenRenderEnv` via the `WrenchObsWrapper`.
- **Replay original LIBERO demonstrations and save force-torque data (HDF5Replayer)**: Replays full trajectories based on LIBERO benchmark demonstrations (e.g., LIBERO-90), reading and saving force-torque data into new HDF5 files.
- **Physical domain shift (PhysicsHelper)**: Modifies physical properties like gripper stiffness, friction, and gravity in Robosuite, and evaluates model robustness against physical domain shifts using the `LiberoForceSocketEvaluator`.
- **A training&testing example**: Demonstrates training and testing of a diffusion policy with added force-torque inputs, evaluating performance under various domain shifts after in-domain training.


## FDP + RenderNet training

FDP 与 RenderNet 蒸馏训练入口、配置及依赖已包含在本项目中，使用方法见
[`TRAINING.md`](TRAINING.md)。统一入口为 `train_pipeline.py`，训练环境依赖为
`requirements-training.txt`，支持单卡、多卡及两阶段自动训练。

## FDP + CAD-WM inference

统一推理接口为 [`InferenceBase`](inference/inference_base.py)，当前实现为 [`FDPInference`](inference/fdp_inference.py)。后续 DP、pi0、pi0.5 可分别继承父类，复用观测、episode reset 和动作返回接口。FDP 逐帧接收图像与 q/dq/tau，未来 wrench 来自 CAD-WM。

模型、数据、输出路径集中在 [`configs/inference/fdp_cad_wm.yaml`](configs/inference/fdp_cad_wm.yaml)。默认 DP 指向本次 6 万步权重；CAD-WM 尚未训练，先保留 `cad_wm.checkpoint: null`，训练后填入 `.pt` 路径。

```bash
conda activate fdp
python run_inference.py --config configs/inference/fdp_cad_wm.yaml --check-config
# 填好 CAD-WM 权重后，回放传感器输入并保存位姿预测：
python run_inference.py --config configs/inference/fdp_cad_wm.yaml
```

入口输出绝对末端位姿 xyz+xyzw（不含夹爪），默认不读取真实 wrench，也不执行仿真器或机器人动作。时间对齐、在线调用和测试说明见 [inference/README.md](inference/README.md)。

## 🛠️ Installation

`LIBERO-FT` shares most of its environment with the original [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO?tab=readme-ov-file#installtion), with only a few additional dependencies required for debugging and training purposes. To get started, follow these steps:

```shell
# Copied from original LIBERO
conda create -n libero python=3.8.13
conda activate libero
git clone https://github.com/ygtxr1997/LIBERO-FT.git  # different here
cd LIBERO-FT
pip install -r requirements.txt
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113

# Optional: only needed by the force benchmark / debug notebooks.
# The q/dq/tau replay script does not import RoboKit.
# conda install -c conda-forge cmake -y
# pip install git+https://github.com/ygtxr1997/RoboKit.git@6066dbb
```

For the q/dq/tau replay script, Python 3.10 is a practical choice for the
listed package versions. A non-interactive local setup is:

```shell
conda create -n libero_ft python=3.10 -y
conda activate libero_ft
cd /path/to/LIBERO-FT
pip install -r requirements.txt
pip install -e .
# Optional for the force benchmark / debug notebooks. Its egl_probe extension
# needs CMake; install it first if you need RoboKit:
# conda install -c conda-forge cmake -y
# pip install git+https://github.com/ygtxr1997/RoboKit.git@6066dbb

# Keep LIBERO's paths inside the checkout; this avoids the first-import prompt.
export LIBERO_CONFIG_PATH="$PWD/.libero"
mkdir -p "$LIBERO_CONFIG_PATH"
cat > "$LIBERO_CONFIG_PATH/config.yaml" <<EOF
benchmark_root: $PWD/libero/libero
bddl_files: $PWD/libero/libero/bddl_files
init_states: $PWD/libero/libero/init_files
datasets: $PWD/dataset
assets: $PWD/libero/libero/assets
EOF

# On a headless machine, MuJoCo offscreen rendering may need:
export MUJOCO_GL=egl
```

The replay script creates the LIBERO config automatically if it does not
exist, but setting `LIBERO_CONFIG_PATH` explicitly makes the dataset and asset
locations reproducible.


## 💽 Datasets

Our replayed trajectories with **wrench force-torque sensory data** injected: [https://huggingface.co/datasets/ygtxr1997/LIBERO-FT](https://huggingface.co/datasets/ygtxr1997/LIBERO-FT)

Original LIBERO data without force-torque data: [https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets](https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets)


## 🚀 Usage

### 1. 🎬 Replay original LIBERO trajectory and save force-torque sensory data

We provide a class `libero.force.hdf5_replayer.HDF5Replayer` to replay LIBERO trajectories online on your own GPU.
`HDF5Replayer` can construct the full environment according to the meta info saved in LIBERO HDF5 files, replay actions frame-by-frame, and consequently check if the replayed trajectory succeds or not.

A usage example for `HDF5Replayer` is available at: [notebooks/force_hdf5_replayer.ipynb](./notebooks/force_hdf5_replayer.ipynb).

#### Replay q and save dq / tau

For position-only replay, use [`scripts/replay_q_mujoco.py`](./scripts/replay_q_mujoco.py). It reads `obs/joint_states` by default, restores the BDDL scene from the HDF5 metadata, copies the original HDF5, and adds exactly two datasets: `data/<demo>/obs/dq` and `data/<demo>/obs/tau`.

```shell
python scripts/replay_q_mujoco.py \
  --dataset /path/to/task_demo.hdf5 \
  --demo demo_0
```

For example, to process demos 10 through 19 (inclusive) from a task and put
the copied dataset with the two new fields under `dataset/wm_data/`:

```shell
python scripts/replay_q_mujoco.py \
  --dataset dataset/libero_90/KITCHEN_SCENE1_open_the_top_drawer_of_the_cabinet_and_put_the_bowl_in_it_demo_wrench.hdf5 \
  --demo-range 10 19 \
  --overwrite
```

The default output is
`dataset/wm_data/<input-name>_with_dq_tau.hdf5`. All original demos and
datasets are preserved; only the selected demos receive
`data/demo_x/obs/dq` and `data/demo_x/obs/tau`. Use `--wm-data-dir` to choose a
different output directory or `--output` to choose the complete output path.
If the output already exists, add `--overwrite` to replace it.
The range endpoints are inclusive. Use `--demo all` for every demonstration or
`--q-key qpos` for another HDF5 key. The default torque is
`data.qfrc_inverse`, which is defined for a q trajectory.
`--tau-source actuator --ctrl-key ctrl` records actuator torque when the
dataset also contains MuJoCo controls. Visualization is disabled by default;
`--video` or `--plot` enables it explicitly. `--xml /path/to/model.xml` can be
used for standalone NPZ/NPY input.

The original LIBERO-FT `obs/wrenches` values are MuJoCo sensor readings, not
measurements affected by video playback speed. The official
[`HDF5Replayer`](./libero/force/hdf5_replayer.py) replays one action per
environment control tick (the example datasets use `control_freq=20`) and
reads `gripper0_force_ee` and `gripper0_torque_ee` from
`sim.data.sensordata`. The q replay script preserves an existing
`obs/wrenches` dataset; it does not regenerate it. Regenerating wrench values
requires the same q/qvel/qacc timeline, model, contacts, and control period.

#### Rerun visualization

Install the Rerun SDK from the active environment:

```shell
python -m pip install rerun-sdk
```

Then visualize two camera streams and the six wrench channels (`Fx`, `Fy`,
`Fz`, `Mx`, `My`, `Mz`):

```shell
python scripts/visualize_hdf5_rerun.py \
  --dataset dataset/libero_90/KITCHEN_SCENE1_open_the_top_drawer_of_the_cabinet_and_put_the_bowl_in_it_demo_wrench.hdf5 \
  --demo 0
```

The script reads `obs/agentview_rgb`, `obs/eye_in_hand_rgb`, and
`obs/wrenches`. It uses the HDF5 `env_args.env_kwargs.control_freq` for the
Rerun time axis; the example dataset is 20 Hz (`dt=0.05 s`). The default mode
logs the complete demo as quickly as possible so it can be scrubbed in Rerun.
Use `--realtime` for wall-clock playback, `--speed 2` for twice-speed playback,
or save a portable recording without opening a viewer:

```shell
python scripts/visualize_hdf5_rerun.py \
  --dataset dataset/libero_90/KITCHEN_SCENE1_open_the_top_drawer_of_the_cabinet_and_put_the_bowl_in_it_demo_wrench.hdf5 \
  --demo 0 --save /tmp/libero_demo_0.rrd --no-spawn
rerun --open /tmp/libero_demo_0.rrd
```

Use `--demo all` or `--demo-range 0 55` to log multiple demos. If the camera
image orientation is already correct in your file, add `--no-flip-vertical`.

To inspect the actual MuJoCo scene interactively while the q trajectory is
being replayed, add `--viewer`:

```shell
python scripts/replay_q_mujoco.py \
  --dataset dataset/libero_90/KITCHEN_SCENE1_open_the_top_drawer_of_the_cabinet_and_put_the_bowl_in_it_demo_wrench.hdf5 \
  --demo 0 \
  --q-key states \
  --q-format state \
  --viewer \
  --overwrite \
  --viewer-hold
```

The viewer uses real-time playback by default. Change only the wall-clock
display speed with `--viewer-speed 2`; this does not change the q sample period,
`dq`, `tau`, or any sensor value. Without `--viewer`, the calculation is
offline and runs as fast as the machine allows. Add `--realtime` if a headless
run must wait one q period between samples. Close the viewer to stop a range
replay; the currently unfinished demo is skipped, while completed demos remain
in the output HDF5.
Collision geometry is hidden in the viewer by default. The green/blue arm
shapes in the MuJoCo viewer are debug collision geoms, not the robot visual
mesh; use `--show-collision` only when you need to inspect contacts. Hiding
them changes only viewer visibility, not collision detection or torque
calculation.
Using `--q-key states --q-format state` replays the complete MuJoCo state,
including object poses, so it is the recommended mode for checking whether a
task scene follows the demonstration. The state parser handles robosuite's
`[time, qpos, qvel, ...]` layout; the leading time value is not treated as a
joint coordinate. The default `obs/joint_states` mode replays the robot joint
positions while keeping non-robot state at its initial value.
For HDF5 input, the script reconstructs the scene from BDDL by default, which
uses the installed LIBERO and robosuite assets consistently. The stored
`model_file` can be selected with `--scene-source dataset` when its asset paths
are available and you need to reproduce that exact XML.
The viewer requires a graphical display. On a remote machine, use X11
forwarding or a desktop session; `MUJOCO_GL=egl` is suitable for offscreen
rendering but does not create a GUI window.

Below are the visualization examples of 2x speedup replayed trajectories (left to right are: original observation saved in LIBERO HDF5, online observation during replay, 6D force-torque sensory data):

<div align="center">
  <img src="./images/replay_00_rst=-1_suc=1_KITCHEN_SCENE1_open_the_top_drawer_of_the_cabinet_and_put_the_bowl_in_it.bddl.gif" width="100%" alt="replay_k1_open_put">
  <p>(Click to play) Based on LIBERO-90: KITCHEN_SCENE1_open_the_top_drawer_of_the_cabinet_and_put_the_bowl_in_it</p>
</div>

<div align="center">
  <img src="./images/replay_00_rst=-1_suc=1_KITCHEN_SCENE7_open_the_microwave.bddl.gif" width="100%" alt="replay_k1_open_put">
  <p>(Click to play) Based on LIBERO-90: KITCHEN_SCENE7_open_the_microwave</p>
</div>


### 2. 🔍 Check the newly saved HDF5 (with force-torque sensory data)

Structure of new HDF5 data saved by `HDF5Replayer`, where a new key named `data/demo_9/obs/wrenches` is added into it like this:

```shell
[GRP ] data/demo_9
[DSET] data/demo_9/actions shape=(201, 7) dtype=float64
[DSET] data/demo_9/dones shape=(201,) dtype=uint8
[GRP ] data/demo_9/obs
[DSET] data/demo_9/obs/agentview_rgb shape=(201, 128, 128, 3) dtype=uint8
[DSET] data/demo_9/obs/ee_ori shape=(201, 3) dtype=float64
[DSET] data/demo_9/obs/ee_pos shape=(201, 3) dtype=float64
[DSET] data/demo_9/obs/ee_states shape=(201, 6) dtype=float64
[DSET] data/demo_9/obs/eye_in_hand_rgb shape=(201, 128, 128, 3) dtype=uint8
[DSET] data/demo_9/obs/gripper_states shape=(201, 2) dtype=float64
[DSET] data/demo_9/obs/joint_states shape=(201, 7) dtype=float64
[DSET] data/demo_9/obs/wrenches shape=(201, 6) dtype=float32
[DSET] data/demo_9/replay_reset_every shape=(1,) dtype=int8
[DSET] data/demo_9/rewards shape=(201,) dtype=uint8
[DSET] data/demo_9/robot_states shape=(201, 9) dtype=float64
[DSET] data/demo_9/states shape=(201, 47) dtype=float64
```
The shape (201, 6) means that the length of a single trajectory is 201 and the dimension of wrench mounted force-torque sensory data is 6.

### 3. 🧠 Train your own force-conditioned action policy

For training and evalution, we have adoped [Diffusion Policy](https://github.com/real-stanford/diffusion_policy) as the baseline and added extra force modality input like this:

```yaml
image_shape: &image_shape [3, 128, 128]  # ori:[3, 240, 320]
shape_meta: &shape_meta  # [WARNING]: the order of shape_meta is very important!!!
  obs:
    image:  # static
      shape: *image_shape
      type: rgb
    gripper:  # gripper
      shape: *image_shape
      type: rgb
    joint_state:
      shape: [6]
      type: low_dim
    force:  # will load `force_torque` from dataset
      shape: [6]
      type: low_dim
```

We named this force-conditioned diffusion policy as `DP-Force`.
The detailed evaluation results of `DP-Force` are available at: [Benchmark Results](#benchmark-results)


### 4. ⚖️ Evaluate your policy under in-domain and force-related domain shift settings

#### Simulator with force-torque observation

This repo provides a class `libero.force.force_benchmark.SingleForceEnv` for force-conditioned robotic manipulation evaluation.

In `SingleForceEnv.init_env()` function, a force-torque wrapper `WrenchObsWrapper` takes as input the original LIBERO env `OffScreenRenderEnv` and inject force-torque sensory data into it.

Below is the init function of `SingleForceEnv`.
```python
class SingleForceEnv:
    def __init__(
            self,
            hdf5_read_path: str = None,
            camera_hw: tuple = (128, 128),
            env_seed: int = 0,
            libero_code_root: str = "/home/geyuan/code/LIBERO-FT/",
            bddl_path: str = None,
            benchmark_task_suite_name: str = None,  # in `libero_{10/90/object/goal/spatial}`
            # PhysicsHelper params
            phy_keywords: tuple = ("cabinet", "drawer", "door", "basket", "object", "tool"),
    ):
```
* `hdf5_read_path`: the force-torque injected HDF5 file generated by `HDF5Replayer`
* `phy_keywords`: for **In-domain** evalutaion, this can be set as `None`; for **Domain-shifted** evaluation, the env will modify the Mujoco physical attributes of the objects (or robotic gripper) whose names contain the keywords

#### Force-related domain shift

The init function of `SingleForceEnv` contains an instance named `self.physics_helper` initialized like this:

```python
self.physics_helper = PhysicsHelper(
            self.env.env.sim.model,
            object_keywords=phy_keywords,
            random_seed=self.env_seed,
        )
# Note: usage: apply_dynamics_shift; restore_original_params;
```

And the `apply_dynamics_shift` of `PhysicsHelper` can modify the Mujoco physical attributes (usually for making task more difficult):
```python
self.physics_helper.apply_dynamics_shift(
            model=self.env.env.sim.model,  # NOTE: don't know why, but we have to resend the sim.model here
            frictionloss_scale=frictionloss_scale,
            damping_scale=damping_scale,
            sliding_friction_scale=sliding_friction_scale,
            gravity_z_range=gravity_z_range,
            tweak_solref_range=tweak_solref_range,
            rng=rng,
        )
```
* `frictionloss_scale/damping_scale/sliding_friction_scale`: controlling the **friction** of the robotic gripper and other interacted objects, larger scale means larger friction for objects and vice versa; for robotic gripper, we always *reduce* the friction to make it smoother and difficult to do contact-relaed tasks.   
* `gravity_z_range=gravity_z_range`: controlling the z-axis of the **gravity**, larger value means heavier.
* `tweak_solref_range`: controlling the **stiffness** of the gripper, larger value means harder (like smoother surface), and smaller value means softer (Note: too small value may leads to *Model Clipping*).
* `rng`: numpy random generator

> **[Important Note]**: 
> 
> While our initial design aimed for a linear correlation between parameter scaling and task difficulty (e.g., enlarging friction scale values to universally increase difficulty), experiments revealed inconsistent effects across tasks; for instance, reduced friction complicates microwave opening (slippage) but simplifies drawer sliding (reduced resistance).
> 
> To address this without manual per-task tuning, we adopt a worst-case evaluation strategy: we measure success rates under opposing physical shifts (e.g., high vs. low friction/stiffness) and report the minimum performance. 
This approach is reasonable, ensuring the resulting evaluation data is meaningful and rigorously reflects the model's true robustness.

#### Socket-based evaluation

Since the python environments for model training and simulator evaluating are usually inconsistent, we also provide a socket-based evaluator (based on [Uvicorn](https://uvicorn.dev/)) named `libero.force.force_benchmark.LiberoForceSocketEvaluator`. A usage example is available at [eval_socket.py](./eval_socket.py).


## 🏆 Benchmark Results

We evaluated the baseline model `DP-Force` (Diffusion Policy with force modality) across 12 selected contact-dependent tasks. 
For each irrelevant task, we train a single DP-force model (12 tasks corresponds to 9 models).
The evaluation protocol tests the model's zero-shot robustness under various physical domain shifts: **In-domain** (`base`), **Friction** scale (`f2`, `f0.5`, and the worst-case `fmin`), **Stiffness** scale (`s50`, `s0.5`, and the worst-case `smin`), and **Gravity** scale (`g2`). 

As shown in the table below, while the model achieves a solid average success rate of 72.83% in-domain, its performance degrades significantly under unseen physical variations. Notably, heavier gravity (`g2`) and worst-case friction shifts (`fmin`) cause the sharpest drops in performance (down to 41.00% and 57.83%, respectively), highlighting the challenging nature of zero-shot transfer in contact-rich environments.


| Task_ID | Ckpt | base | f2 | f0.5 | s50 | s0.5 | g2 | fmin | smin |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **k10_close_put_blackbowl** | ep550 | 56 | 44 | | 52 | 62 | 22 | 44 | 52 |
| **k1_open_top** | ep600 | 84 | 74 | | 72 | 88 | 48 | 74 | 72 |
| **k5_close** | ep1350 | 80 | 88 | 74 | 76 | 76 | 46 | 74 | 76 |
| **k10_close** | ep550 | 98 | 98 | | 96 | 94 | 44 | 98 | 94 |
| **k2_open** | ep1350 | 88 | 56 | | 88 | 80 | 66 | 56 | 80 |
| **k6_close** | ep600 | 56 | 50 | 70 | 54 | 70 | 22 | 50 | 54 |
| **k1_open_bottom** | ep1000 | 82 | 36 | 74 | 70 | 78 | 34 | 36 | 70 |
| **k4_close_bottom_open_top** | ep1400 | 46 | 28 | | 42 | 56 | 18 | 28 | 42 |
| **k7_open** | ep700 | 70 | 76 | 56 | 72 | 46 | 34 | 56 | 46 |
| **k1_open_top_put_bowl** | ep600 | 56 | 36 | | 52 | 66 | 28 | 36 | 52 |
| **k4_close_bottom** | ep1400 | 98 | 94 | | 94 | 96 | 88 | 94 | 94 |
| **s3_pick_book_place_leftcaddy** | ep850 | 60 | 48 | 54 | 50 | 50 | 42 | 48 | 50 |
| **Averaged Success Rate (%)** |  | **72.83** | 60.67 | 65.60 | 68.17 | 71.83 | **41.00** | **57.83** | **65.17** |

The map between `Task_ID` and original HDF5 filename can be found at: [documents/LIBERO_task_id.md](./documents/LIBERO_task_id.md).


#### Ablation Study: Is force sensor helpful?

To verify the effectiveness of the injected force-torque observations, we conducted an ablation study by masking out the force data (`force*0`). The table below compares the standard `DP-Force` model against the ablation baseline on two representative tasks. 

| Model / Setting | Task_ID | Ckpt | base | f2 | g2 | s50 | Shift Avg |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **DP-Force** | k1_open_top | ep600 | **84** | **74** | **48** | 72 | **64.67** |
| *DP w/o Force* | k1_open_top | ep700 | 78 | 66 | 36 | **74** | 58.67 |
| **DP-Force** | k1_open_top_put_bowl | ep600 | **56** | **36** | **28** | **52** | **38.67** |
| *DP w/o Force* | k1_open_top_put_bowl | ep700 | 52 | **36** | 26 | 44 | 35.33 |

As shown in the `Shift Avg` column, the model equipped with force-torque sensors (`DP-Force`) achieves consistently higher average success rates under physical domain shifts compared to the baseline without force inputs. This demonstrates that incorporating the force modality not only maintains in-domain performance but, more importantly, enhances the model's robustness and adaptability to unexpected, unseen physical dynamics.


## 🗂️ Project Structure

```shell
LIBERO-FT/
├── libero/
│   ├── force/               # Core code for force-torque injection & domain shifts
│   │   ├── ...              # (Contains HDF5Replayer, PhysicsHelper, etc.)
│   └── datasets/            # Recommended directory for storing HDF5 datasets
├── notebooks/               # Jupyter notebooks for usage examples
├── eval_socket.py           # Socket-based evaluation script 
├── requirements.txt         # Dependencies
└── README.md                # Project documentation
```

## ✅ TODO List

- [ ] **Release DP-Force pipeline**: Release the complete training and testing codebase for the `DP-Force` baseline.
- [ ] **Scale up dataset**: Process and inject force-torque data for the remaining tasks in the LIBERO-90/130 datasets.
- [ ] **More Baselines**: Evaluate multi-task and language-conditioned foundation models.
- [ ] **Dynamic Domain Shifts**: Implement intra-episode physical parameter changes (e.g., objects becoming slippery dynamically during the operation).
- [ ] **New contact-rich tasks**: Design and integrate genuine contact-rich tasks (e.g., screwdriving, wiping glass, sweeping) from scratch.


## 🚧 Limitations

- **Lack of True Contact-Rich Tasks & Data Scale:** The original LIBERO benchmark does not provide dedicated contact-rich tasks (e.g., screwdriving, wiping glass, sweeping). Constrained by computational resources, we have currently only processed 11 open/close tasks and 1 pick-and-place task from LIBERO-90, as these still heavily rely on force feedback. We will continue processing more data and highly encourage the community to contribute by sharing the processing workload or designing genuine contact-rich tasks from scratch.
- **Limited Baselines:** We only benchmarked Diffusion Policy with additional force modality input, which is nominally a single-task model without natural language conditioning support. We plan to evaluate more multi-task and language-conditioned baselines in the future and welcome researchers to share their own benchmarking results on our dataset.
- **Simplified Domain Shifts:** The current implementation of force-related domain shifts is relatively simple and may not be comprehensive enough. For instance, physical parameters cannot dynamically change over time within an episode, and we have not fine-tuned shift parameters for each individual task.


## 🤝 Contributing

We welcome contributions to `LIBERO-FT`! If you'd like to contribute, please fork the repository and submit a pull request.

## 🎓 Citation

```latex
@misc{LIBEROFT2026,
  author = {{Yuan, Ge and Xu, Dong}},
  title = {{LIBERO-FT: Evaluating Robotic Manipulation under Force-Related Domain Shifts}},
  year = {2026},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/ygtxr1997/LIBERO-FT}},
}
```

## ❤️ Acknowledgements

- [Mujoco](https://mujoco.org/) for the underlying force sensor simulation.
- [Robosuite](https://robosuite.ai/) for providing the robotic simulation environment.
- [LIBERO Benchmark](https://libero-benchmark.com/) which this codebose highly depends on.
