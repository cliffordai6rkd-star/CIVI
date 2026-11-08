# FDP + CAD-WM 推理

父类位于 `inference_base.py`，FDP 实现位于 `fdp_inference.py`。后续 DP、pi0、pi0.5 可以各自继承 `InferenceBase`，实现 `spec`、`reset`、`observe`、`predict`；共用 `InferenceObservation` / `InferenceResult`，由各子类声明所需字段与动作格式。当前仅 FDP 已实现。

所有权重、仓库、数据和输出路径由 [配置](../configs/inference/fdp_cad_wm.yaml) 控制。路径相对配置中的 `project_root` 解析，因此可以从其他工作目录启动。默认使用 2026-11-05 蒸馏流水线的教师 `optimizer_step=00180000.ckpt`（EMA 和保存的 normalizer），以及与 DP 原始 wrench 数据同源的 `cad-wm/outputs/cad_wm_median3_200k/checkpoints/step_00180000.pt`。世界模型每帧由 q/dq/tau 生成未来 wrench，教师使用原始双视角 RGB 与预测 wrench 输出末端位姿；这条入口不加载 RenderNet。

使用 `configs/inference/fdp_teacher_student_compare.yaml` 可以比较冻结
teacher 在原始 RGB 与 RenderNet RGB 上的动作差异：

```bash
python -m inference.compare_teacher_student \
  --config configs/inference/fdp_teacher_student_compare.yaml
```

脚本固定同一 CAD-WM 预测和同一扩散随机状态。teacher 使用原图和 wrench
融合表征；student 使用 `RenderNet → encode_image_context → DiT`，不再调用
teacher 的 wrench 编码和融合层，与蒸馏训练路径一致。可复用学生预测函数为
`inference.render_student.predict_student_action`，返回同 FDP 的 action 字典。
脚本保存 `teacher_actions`、`student_actions`、逐动作差值及统计量。输入是
LIBERO-FT 的 HDF5 记录回放；它不执行仿真器或机器人动作，因此结果是离线
动作影响分析，不是成功率或真机验证。配置默认比较 `unrestricted`，将
`student.checkpoint` 改为 `context_p90/checkpoints/best.ckpt` 并打开
`use_attention_mask` 即可比较另一版。

**旧结果作废：** 2026-11-04 首次生成的 `outputs/inference/teacher_student_*`
离线报告错误地把 rendered RGB 再送入完整 teacher wrench fusion，未执行
训练中的 student 图像单独编码路径。旧 action-delta 数值及据此得出的
unrestricted / context_p90 排名不能用来评价 student。原 NPZ 保留，旁边 JSON
标为 `status: INVALIDATED` 并记录原因，详情见该输出目录的
`INVALIDATED_teacher_student_results.md`。修复后的结果包含
`student_path_version: 2`；这些离线结果仍不代表仿真任务成功率。

```bash
cd /home/lcy/code/LIBERO-FT
conda activate fdp

# 当前即可运行：只核对两个训练配置的采样率、时间窗与结构，不加载权重。
python run_inference.py --config configs/inference/fdp_cad_wm.yaml --check-config

# 使用两个训练完成的 checkpoint，回放 demo_0 的前 100 帧并输出预测。
python run_inference.py --config configs/inference/fdp_cad_wm.yaml
```

默认输出为 `outputs/inference/fdp_cad_wm_teacher_180k_raw_wm_demo_0.npz`，同时保存
JSON 报告和解析后的配置。再次运行前请修改 `replay.output`，入口会拒绝覆盖
已有结果。运行需要 `fdp` 环境及 CUDA；模型接口也可以使用配置中的 CPU 设备。

此前 `outputs/inference/fdp_cad_wm_teacher_200k_demo_0.*` 是旧教师 checkpoint
的历史回放结果，切换到 18 万步教师后需要重新运行回放验证；离线回放仍不等同于
仿真成功率。

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
    wrench=wrench6,  # 当前实测 [Fx,Fy,Fz,Mx,My,Mz]；默认 measured 模式必填
)
result = engine.predict(observation)
if result.ready:
    pose_chunk = result.actions             # [8,8]，xyz + xyzw + gripper
    pose_times = result.action_timestamps_s  # 每个参考位姿的时间
else:
    print(result.info["warmup_reason"])

# 若控制循环只在部分帧重规划：每帧 engine.observe(observation)，
# 需要新动作时 engine.predict()；不要重复提交同一时间戳。
```

每个 CAD-WM 调用只读取截至当前帧的 q/dq/tau 历史，按其检查点归一化，产生未来 `t+[0.05,...,0.40]` 秒的 8×6 wrench，反归一化为传感器坐标系下的 N / N.m 后再交给 FDP 自己的归一化与接触预处理。WM 的采样率、未来时间点、7 轴 / 6 通道格式必须与 FDP 完全匹配；不做隐式重采样或坐标变换。

FDP 的两帧图像各配自己的未来预测。上一帧的预测保存在缓存中，不用当前预测覆盖它。默认配置使用 `phase_wrench_source: measured`，用当前实测 wrench 驱动迟滞接触状态；未来网络 wrench 仍只由 CAD-WM 产生。在线配置的 `future_contact_mode: legacy_rollout` 使用同一接触检测器的进入/退出迟滞，避免随机未来预测在 8 N 边界反复切换图像-only 与力觉融合分支。设置为 `cad_wm` 时则使用上一时刻预测的第一个点作为当前 wrench 估计；第一帧没有因果估计，随后累计两帧有效图像上下文，因此允许历史 padding 的 WM 默认前两帧返回预热，第三帧开始输出动作。短历史重复最早有效值，episode reset 清空全部缓存与随机数生成器。若 WM 训练禁止 history padding，则等待完整历史后再预热上下文。

若使用 `phase_wrench_source: measured`，需要额外传入当前 `wrench=[Fx,Fy,Fz,Mx,My,Mz]`；它只用于历史接触状态，网络未来 wrench 始终由 CAD-WM 产生。`future_contact_mode` 的默认值为空时沿用 checkpoint 的预处理模式，显式设置为 `legacy_rollout` 才启用部署迟滞。

当前默认回放还读取 HDF5 的 `obs/wrenches` 作为接触迟滞输入；切回 `phase_wrench_source: cad_wm` 后可只使用图像、q/dq/tau 和时间。`predict_every` 只改变 DP 重规划频率，WM 仍处理每个 20 Hz 帧。结果保存在 `replay.output` 指定的 NPZ，以及旁边的 JSON 和解析后配置；已有 NPZ 会报错，避免覆盖实验。

当前 DP 检查点的训练目标是绝对 `observation.ee_pose + action[:,6]`，输出 8D 位姿和夹爪命令；教师的显式观测只有双视角 RGB、历史 wrench 和未来 wrench。FDP 适配器仍兼容只有 7D 位姿输出的旧 checkpoint，在线服务会把夹爪列附加到 6D OSC 命令上。模型返回的动作索引从当前 anchor 开始，时间为 `t+[0,...,0.35]` 秒；wrench 预测时间为 `t+[0.05,...,0.40]` 秒，入口分别标注。回放不会调用 `env.step`；仿真/硬件执行端需要显式的末端位姿控制器。

若要在同一条命令里启动 FDP/CAD-WM、执行 MuJoCo 推理并查看 wrench，使用根目录的 `eval_socket.py --interface fdp --start-fdp-server --wrench-plot`。窗口中的实线是 MuJoCo 当前实测 wrench，虚线是同一帧送入 DP 的 CAD-WM 八点未来预测；不需要再单独运行 `run_inference.py`。

CAD-WM 训练数据中的 tau 是 MuJoCo 逆动力学值，离线 dq/tau 可能含后续帧信息。回放只验证接口与时间窗口；在线接入时应提供与训练定义、单位和坐标系一致的因果状态/力矩输入。当前不会从未来 q 差分或用真实未来 wrench 补齐缺失输入。

CAD-WM 运行时使用私有包名导入，避免其 `train` 包与 DP 仓库的 `train.py` 冲突；两个模型都使用 eval 和 inference_mode，不加载优化器。`reset(seed=...)` 重置两个独立的模型采样随机数生成器。
