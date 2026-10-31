# FDP + CAD-WM 推理

父类位于 `inference_base.py`，FDP 实现位于 `fdp_inference.py`。后续 DP、pi0、pi0.5 可以各自继承 `InferenceBase`，实现 `spec`、`reset`、`observe`、`predict`；共用 `InferenceObservation` / `InferenceResult`，由各子类声明所需字段与动作格式。当前仅 FDP 已实现。

所有权重、仓库、数据和输出路径由 [配置](../configs/inference/fdp_cad_wm.yaml) 控制。路径相对配置中的 `project_root` 解析，因此可以从其他工作目录启动。默认 DP 指向本次 `optimizer_step=00060000.ckpt`，加载 EMA 权重和保存的 normalizer。CAD-WM 尚未训练，`cad_wm.checkpoint` 留空；实际推理前必须填写训练后的 `.pt` 路径。

```bash
cd /home/lcy/code/LIBERO-FT
conda activate fdp

# 当前即可运行：只核对两个训练配置的采样率、时间窗与结构，不加载权重。
python run_inference.py --config configs/inference/fdp_cad_wm.yaml --check-config

# 填好 cad_wm.checkpoint 后，使用 replay 配置回放一段传感器数据并输出预测。
python run_inference.py --config configs/inference/fdp_cad_wm.yaml
```

在线调用的公共接口：

```python
from inference import InferenceObservation
from inference.fdp_inference import FDPInference

engine = FDPInference.from_config("configs/inference/fdp_cad_wm.yaml")
engine.reset(seed=42)  # 每个 episode 调一次，不允许跨 episode 复用历史。

# 在每个 20 Hz 传感器帧调用，包括两次动作重规划之间的帧。
observation = InferenceObservation(
    timestamp_s=t,
    images={"wrist": wrist_rgb, "side": side_rgb},  # uint8 HWC RGB
    q=q7, dq=dq7, tau=tau7,  # rad / rad/s / N.m，当前帧，Panda 前七轴
)
result = engine.predict(observation)
if result.ready:
    pose_chunk = result.actions             # [8,7]，xyz + xyzw
    pose_times = result.action_timestamps_s  # 每个参考位姿的时间
else:
    print(result.info["warmup_reason"])

# 若控制循环只在部分帧重规划：每帧 engine.observe(observation)，
# 需要新动作时 engine.predict()；不要重复提交同一时间戳。
```

每个 CAD-WM 调用只读取截至当前帧的 q/dq/tau 历史，按其检查点归一化，产生未来 `t+[0.05,...,0.40]` 秒的 8×6 wrench，反归一化为传感器坐标系下的 N / N.m 后再交给 FDP 自己的归一化与接触预处理。WM 的采样率、未来时间点、7 轴 / 6 通道格式必须与 FDP 完全匹配；不做隐式重采样或坐标变换。

FDP 的两帧图像各配自己的未来预测。上一帧的预测保存在缓存中，不用当前预测覆盖它。FDP 的历史 wrench 用于当前接触判定，默认 `phase_wrench_source: cad_wm`：上一时刻预测的第一个点，成为当前时刻的 wrench 估计。这个分支完全不读取真实 wrench；第一帧没有因果估计，随后累计两帧有效图像上下文，因此允许历史 padding 的 WM 默认前两帧返回预热，第三帧开始输出动作。短历史重复最早有效值，episode reset 清空全部缓存与随机数生成器。若 WM 训练禁止 history padding，则等待完整历史后再预热上下文。

可选择 `phase_wrench_source: measured`，此时额外传入当前 `wrench=[Fx,Fy,Fz,Mx,My,Mz]`，仅历史接触判定使用测量值；网络未来 wrench 始终由 CAD-WM 产生。预测得到的相位估计与训练时的真实历史力存在分布差异，最终任务表现需要独立闭环评估。

默认回放仅逐帧读取 HDF5 的图像、q/dq/tau 和时间，不读取 `obs/wrenches`。可以删除 HDF5 中的真实 wrench，默认分支仍可回放。`predict_every` 只改变 DP 重规划频率，WM 仍处理每个 20 Hz 帧。结果保存在 `replay.output` 指定的 NPZ，以及旁边的 JSON 和解析后配置；已有 NPZ 会报错，避免覆盖实验。

该 DP 检查点的训练目标是绝对 `observation.ee_pose`，不是原始 OSC_POSE 动作，且没有夹爪输出。模型返回的动作索引从当前 anchor 开始，时间为 `t+[0,...,0.35]` 秒；wrench 预测时间为 `t+[0.05,...,0.40]` 秒，入口分别标注。回放不会调用 `env.step`；仿真/硬件执行端需要显式的末端位姿控制器和夹爪策略。

CAD-WM 训练数据中的 tau 是 MuJoCo 逆动力学值，离线 dq/tau 可能含后续帧信息。回放只验证接口与时间窗口；在线接入时应提供与训练定义、单位和坐标系一致的因果状态/力矩输入。当前不会从未来 q 差分或用真实未来 wrench 补齐缺失输入。

CAD-WM 运行时使用私有包名导入，避免其 `train` 包与 DP 仓库的 `train.py` 冲突；两个模型都使用 eval 和 inference_mode，不加载优化器。`reset(seed=...)` 重置两个独立的模型采样随机数生成器。
