# FDP 与 RenderNet 蒸馏训练

本目录包含从 `dp_hirol_main` 提取的训练代码及其 Python/Hydra 依赖，保留
`diffusion_policy` 包路径以兼容已有 checkpoint。文件清单与源文件校验值见
[`documents/fdp_training_manifest.json`](documents/fdp_training_manifest.json)。

## 环境与资源

在本项目根目录运行命令。本机可以直接复用现有环境：

```bash
cd /home/lcy/code/LIBERO-FT
conda activate fdp
export DP_DATASET_PATH=/home/lcy/code/dp_hirol_main/data/libero_open_top_drawer_put_bowl_20hz_lerobot_v3
export DINOV3_MODEL_PATH=/home/lcy/code/dp_hirol_main/model/model/dinov3-vitb16-pretrain-lvd1689m
```

这两个变量可以指向任意位置的 LeRobot v3 数据集和 DINOv3 权重目录。迁移到其他
机器时，先准备这两份资源，再设置它们的实际绝对路径。代码副本约 0.8 MB；数据、
预训练权重、训练输出和缓存单独管理。

新建训练环境时，从项目根目录执行：

```bash
conda env create -f conda_environment_fdp.yaml
conda activate fdp-training
```

`requirements-training.txt` 对应 Python 3.10 / PyTorch 2.7.1 / CUDA 12.8，
FFmpeg 及 C++ 运行库由上述 Conda 文件提供。现有 `requirements.txt` 对应旧版
LIBERO 仿真环境；训练环境使用独立的依赖清单。直接从根目录运行即可加载本地包，
无需重新安装根目录的 LIBERO `setup.py`。

## FDP 训练

默认使用 20 Hz、DINOv3 ViT-B/16、两层 wrench LSTM、一层 image-query
cross-attention、8 层 DiT，batch 128、100,000 个优化步；50 条演示全部用于训练。
接触以 XYZ 力模长 8 N 判定，连续 5 帧超过/低于阈值才切换状态。
图像采用接触相关的模糊课程，sigma 在 3,000–10,000 步从 5 降至 0。

```bash
# 预览两阶段命令，不创建训练目录、不分配 GPU。
python train_pipeline.py all --dry-run

# 单独训练 FDP，关闭 W&B 在线日志。
python train_pipeline.py dp --gpus 0 \
  --dp-override logging.mode=disabled \
  --output-dir data/outputs/my_fdp

# FDP 完成后，自动选择教师 checkpoint 并开始 RenderNet 蒸馏。
python train_pipeline.py all --gpus 0 \
  --dp-override logging.mode=disabled \
  --output-dir data/outputs/my_fdp_and_distill

# 四卡训练：batch 按全局大小分配，每卡放一份模型。
python train_pipeline.py all --gpus 0,1,2,3 --nproc-per-node 4 \
  --dp-override logging.mode=disabled \
  --output-dir data/outputs/my_fdp_and_distill_ddp
```

统一入口的 `--output-dir` 必须是新目录，两个阶段分别写入其 `dp/` 与 `distill/`。
默认保存最近三个周期 checkpoint，每 20,000 个优化步保存一次。

原始 wrench-only 配置仍可直接运行：

```bash
python sim_force_aware_dp_train.py \
  --config diffusion_policy/config/sim_force_aware_dp_train_20hz_lstm2.yaml \
  logging.mode=disabled
```

真实机器人 `tau_ext` 接触门控及 USB image-query 教师的配置也已包含，使用
`train.py --config diffusion_policy/config/alation_study/train_dp_image_query_lstm_contact_blur_insert_usb.yaml`
等入口，按相应任务设置 `DP_DATASET_PATH`。

## 单独蒸馏与恢复

```bash
# 也可以给 --teacher-run 传其他项目里的绝对训练目录。
python train_pipeline.py distill --gpus 0 \
  --teacher-run data/outputs/my_fdp/dp \
  --output-dir data/outputs/my_distillation

# 直接指定教师 checkpoint。
python train_distill_render_net.py \
  teacher_checkpoint=/absolute/path/to/teacher/checkpoints/latest.ckpt

# 恢复 RenderNet 本身的优化器、epoch 和 step。
python train_distill_render_net.py \
  teacher_checkpoint=/absolute/path/to/teacher/checkpoints/latest.ckpt \
  training.resume_from=/absolute/path/to/distill/checkpoints/latest.ckpt
```

蒸馏优先读取教师保存的 EMA 权重和 normalizer，网络结构继承教师。可用
`DINOV3_MODEL_PATH` / `DP_DATASET_PATH` 覆盖旧 checkpoint 里的资源路径，
对应 Hydra 参数为 `dino_model_path` / `dataset_path`。若不指定则沿用教师路径。
单独调用蒸馏入口时，也可用 `FDP_TEACHER_CHECKPOINT` 指定默认教师。

教师 token 默认以 float16 保存在本项目 `data/cache/render_net_teacher/` 的 SSD
缓存中，RGB 解码到 RAM。正常结束、早停、异常和 Ctrl+C/SIGTERM 退出时，自动
删除本次教师 cache；多卡共享时由最后一个使用者清理。需要保留缓存时追加
`teacher_cache.cleanup_on_exit=false`；`kill -9` 或断电无法执行退出清理。

FDP 续训使用保存的完整配置及原输出目录：

```bash
python sim_force_aware_dp_train.py \
  --config /absolute/path/to/run/.hydra/config.yaml \
  training.resume=true logging.mode=disabled \
  dataset_path="$DP_DATASET_PATH" dino_model_path="$DINOV3_MODEL_PATH" \
  task.dataset.dataset_path="$DP_DATASET_PATH" \
  policy.dino_model_name_or_path="$DINOV3_MODEL_PATH" \
  hydra.run.dir=/absolute/path/to/run
```

如需转换原始 LIBERO HDF5，使用 `python scripts/h5_2_lerobotv3.py --help`。
该转换器支持生成训练所需的 LeRobot v3 格式；20 Hz 任务需要用 20 Hz 数据配置。

## 验证与日志

```bash
# 从本项目运行，确认实际加载的包位于这里。
python -c 'import diffusion_policy; print(diffusion_policy.__file__)'
python sim_force_aware_dp_train.py --config-name sim_force_aware_dp_train_20hz_lstm2 --cfg job
python train_distill_render_net.py --cfg job

# 小型 CPU 回归测试，不启动真实数据训练。
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q -p no:cacheprovider \
  tests/test_train_pipeline.py tests/test_force_aware_diffusion.py \
  tests/test_render_net_teacher_cache.py tests/test_render_net_teacher_cache_cleanup.py \
  tests/test_render_net_ddp.py tests/test_render_net_distillation.py

tail -f data/outputs/my_fdp/dp/train.log
python -m diffusion_policy.scripts.plot_render_net_training data/outputs/my_distillation/distill
```
