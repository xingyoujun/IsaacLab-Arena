# G2 工作台清理

`g2_clean_workcell_table`：铝块放蓝框、电钻放黄框、齿轮放绿框，替代原工具整理任务。

## 与 stack bowls 一致的采集流程

1. cuRobo MotionGen 为接近、抓取后的抬升、搬运、下降、撤离分别生成轨迹，直接执行插值关节目标。没有自定义局部 IK 或 5mm/8步细分。
2. 物理抓取，夹爪闭合后检查实际抬升；搬运中的附着球只用于规划碰撞，不焊接、不瞬移物体。接触阶段从规划世界排除当前被抓物体，与 stack bowls 相同；表面接触不再额外逐步调用 CPU 三角网格检查。
3. 沿用 stack bowls 现场采集示例的标准 Arena 相机 recorder，HDF5 保存动作、状态、core 和三路原始 pre-action RGB；失败保留，不进入成功数据集。正式录制前进行一次未记录的物理控制步预热，初始状态从预热后的实测状态开始。
4. 使用同一个 `g2_dataset_io.py`、`g2_raw_to_lerobot.py` 导出三路原生图像和 LeRobot v2.1。直接编码原始 RGB，不进行 CUDA 状态重渲染；当前 Isaac Sim 构建的无物理步 CUDA 状态恢复会导致渲染滞后。原来的 stack bowls CPU 离线渲染分支保持不变。
5. 使用同一个 `check_g2_dataset.py` 检查 parquet、视频帧数和原始数组一致性。

物理仿真、MotionGen 与渲染使用 CUDA；编码使用 NVENC。调度、文件 I/O 和独立数据审计仍使用 CPU。当前为固定布局试采，不代表随机布局成功率。

## 运行

使用 dev-container 技能找到本集成目录容器，以宿主用户在 `/workspaces/isaaclab_arena` 中执行：

```bash
PYTHONPATH=$PWD:$PWD/submodules/IsaacLab/source/isaaclab \
ISAACLAB_ARENA_FORCE_EXIT_ON_COMPLETE=1 \
/isaac-sim/python.sh isaaclab_arena_examples/g2_collect_workcell_dataset.py \
  --root /datasets/g2_workcell_dataset \
  --raw_root /datasets/g2_workcell_raw \
  --demos 1 --max_attempts 3 --stage all
```

`--stage core` 只采集，`--stage render` 只导出。与 stack bowls 相同，配置和源码哈希固定，支持中断恢复，已有输出不覆盖。

```bash
/isaac-sim/python.sh isaaclab_arena_examples/check_g2_dataset.py \
  --root /datasets/g2_workcell_dataset/clean_workcell_table \
  --work_dir /datasets/g2_workcell_raw/clean_workcell_table
```

输出三路：`observation.images.head`、`observation.images.left_wrist`、`observation.images.right_wrist`；16维绝对关节/夹爪动作。相机沿用 G2CameraCfg，视野由真实机器人姿态决定。

模型位于容器 `/datasets/g2_workcell_assets`，供应商配置位于 `/datasets/agibot_dataset_v1_raw/assets/g2/robot.yaml`，不纳入 Git。

## 历史记录

`outputs/g2_workcell_validation_20260925` 与 `outputs/g2_workcell_cuda_20260925` 是改造前的实验记录，不作为本版 MotionGen 采集流程的通过证据。独立 replay 脚本仅用于物理回放诊断，正式数据导出使用上面的统一入口。旧的 CUDA 状态重渲染视频无效，不能作为训练数据。

## 本地试采验证（2026-09-25）

`outputs/g2_workcell_motiongen_20260925/dataset_004/clean_workcell_table` 已在原始 v1 布局完成一条固定布局试采：2514 帧、15 FPS，三类物体全部通过独立物理审计和原始/parquet/三路视频对齐检查。过程与耗时见同目录上级的 `RESULT.md`。官方 LeRobot loader 附加检查因运行环境缺少 `lerobot` 尚未执行。

## data engine 原生 cuMotion 试验

`--planner_backend cumotion` 使用 `isaaclab_arena_cumotion.planner.CumotionArmPlanner` 背后的
Isaac Sim `GraphBasedMotionPlanner`，并复用 `EnvActionExecutor`。G2 仍使用 16 维动作和二值夹爪；
不直接套用 data engine 原 Agibot 的 20 维动作。配置为独立的 `motion_config_cumotion.yaml`，
默认 `curobo` 及已有现场 RGB 采集入口保持原样。

```bash
/isaac-sim/python.sh -m isaaclab_arena_examples.g2_workcell.collect \
  --headless --device cuda:0 --num_envs 1 --stage sequence \
  --planner_backend cumotion --output_dir outputs/my_native_trial

/isaac-sim/python.sh -m isaaclab_arena_examples.g2_workcell.rerender_states \
  --headless --enable_cameras --device cuda:0 --num_envs 1 \
  --raw_dir outputs/my_native_trial --output_dir outputs/my_native_trial_views
```

此入口先保存状态，再生成三路视频旁车文件。补渲染遵循 data engine 的“恢复状态后推进一个物理步”方式，
报告中量化额外物理步造成的末端/物体位姿偏差。它使用 post-action 状态，不能冒充原流程的精确 pre-action RGB，
也尚未接入原来的 LeRobot 成功数据导出入口。

原生后端的碰撞世界使用房间盒体、工具包围盒、每个框的五块壁/底板和持物球覆盖，
与 cuRobo 的场景三角网格表示不同，因此不能把规划调用时差解释成严格的同条件规划器性能对比。
G2 的电钻半圈换向单独允许第 5 关节至多 3.65 rad，其余关节仍限制 2.6 rad，并保留 0.03 rad 限位余量。
若原生位姿查询选择了不合适的肩肘分支，则尝试原生 IK 候选，并再次进行完整关节空间碰撞路径规划。
不合格路径不会执行，实际抓取/放置/稳定性判定不变。

物理及三路渲染使用 CUDA，编码使用 NVENC。原生规划设备由库管理；独立 Nsight 探针中有 CUDA 内存调用，
未观察到 GPU kernel，不能宣称整个图规划计算都在 CUDA 上。实验记录位于 `outputs/g2_workcell_cumotion_20260925`。

2026-09-25 的 `sequence_004` 试跑共 1060 帧：铝块放置完成，电钻搬到黄框上方，
在 `lower_into_bin` 原生规划返回失败后停止，齿轮未执行。独立原始数据审计通过，
但 `task_success=false`；对应三路补渲染仅用于失败诊断，不能进入成功训练数据。
因此原生 cuMotion 仍为实验入口，当前完整任务通过的基线是 cuRobo MotionGen。

## 2026-09-26 场景调整

当前场景规格为 v2：铝块与齿轮线性尺寸缩小 10%，电钻缩小 15%，框壁由约 9.35 cm 降至 7.19 cm。
电钻初始中心改为 `(-0.18, 0.06)` m、朝向 -16°，黄框中心为 `(0.07, 0)` m。
其他物体和框的平面位置保持原配置。抓取坐标同步按尺寸调整；原生 cuMotion 的齿轮预抓取高度为 0.31 m。
缩放同时应用于 USD 实体、规划网格与任务判定，不只修改视觉模型。启动静置后检查所有物体/框的水平偏移小于 5 mm。

离线补渲染从每次采集保存的 `sources/clean_workcell_table.yaml` 构造场景，避免新布局污染旧记录。
原 v1 的 cuRobo 成功结果只证明旧布局通过；新布局的验证结果见 `outputs/g2_workcell_cumotion_20260926/RESULT.md`。

`sequence_003` 已完成新版布局的一条原生 cuMotion 试采：1888 帧、15 FPS，三个物体均完成放置和撤离，
任务成功且独立原始数据审计通过。齿轮撤离在位姿规划选择大幅肩部翻转时，改用当前姿态附近的原生 IK 候选，
再进行完整关节空间碰撞规划；实际最大单关节转动 1.987 rad，限位余量 0.225 rad。
这是一个固定布局/种子的完整样本，尚不代表随机布局成功率；离线图像对齐及原生规划设备的限制仍同上。
