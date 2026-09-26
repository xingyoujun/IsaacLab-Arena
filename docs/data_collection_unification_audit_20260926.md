# G2 / UR7e 数据采集统一：代码差异审核

日期：2026-09-26。状态：**审核与方案，尚未接入生产流程，尚未迁移或重渲染正式数据。**

按本次用户确认，范围为 G2 的 `stack_bowls`、`peg_into_sleeve`，以及 data_engine 的全部 UR7e 任务；排除旧 Agibot 任务。UR7e 原始数据位于另一台机器，后续由用户安排同步。本次只检查本地代码、已有成品元数据与抽样视频，不连接源机器。

> 更新说明：本报告基于合并 `ccefc86a` 之前的代码；其中 UR7e 样本和 12 个 case 属于 RR real2sim，不能当作 Pine WM 的数据或任务覆盖。新提交将 Pine WM 明确独立，并为 first20 增加 pre-step 状态映射与有界微步重渲染。当前范围和后续整合以 [当前 G2 工作流程](g2_development.md) 为准。下文保留为历史证据；其中“不推进物理”的建议是理想目标，尚非已验证可用的统一渲染实现。

## 结论

统一应以“物理状态、图像和动作的时间含义相同”为核心，而不是仅改目录名或补齐数组维度。

对每个控制步 t，统一定义：

```
记录 s_t、图像 I_t、观测 o_t → 执行动作 a_t → 记录 s_(t+1)
```

T 个有效动作对应 T 行 pre-action 观测、T 帧视频、T 行 next-state；保存初始状态时物理状态序列共 T+1 个时间点。时间戳为 `t / control_fps`，所有视频帧必须与对应行的 pre-action 状态一致。不得默认丢弃末行，也不得为了更新渲染而推进物理时间。

可以保留 cuRobo、原生 cuMotion 等不同规划器，但规划器只生成动作，必须经过同一采集接口、原始数据规范和验收流程。G2 和 UR7e 的关节、夹爪、相机数量与标定保留在各自的机器人配置中。

## 本地证据及范围

- G2：`/home/ubuntu/datasets/agibot_dataset_v1` 中两个任务各 200 条，共 468679 帧，15 Hz；原始 HDF5 和采集时源码均在本机。
- UR7e：`/home/ubuntu/datasets/rr_real2sim_data` 中六套成品，各 200 条，共 305009 帧，15 Hz；成品有 LeRobot 与 DP Zarr。原始 HDF5、原始随机化记录和完整资产/运行快照未在该目录中找到。
- 六套为 `usdcraft_open_drawer`、`usdcraft_press_toaster`、`miniworkflow_gptsol_open_drawer`、`articraft_open_drawer`、`miniworkflow_astra_open_drawer`、`articraft_press_toaster`。
- 仓库主流程定义 8 个抽屉/按压任务和 4 个旋钮任务，共 12 个 UR7e case。其余 6 个 case 的本地成品缺席，不能据此判断源机器是否已完成。
- 只读抽查每套成品首、中、末三条，共 24 条：G2 的 6 条均视频帧数=Parquet 行数；UR7e 的 18 条均视频帧数=Parquet 行数+1。帧数来自 ffprobe 容器元数据；本次没有完整解码所有视频。
- 结果在 `outputs/unified_collection_20260926/export_sample_audit.json`。这是抽查，不是 1600 条数据的完整验收。
- UR7e 的实际原始时间关系、采集版本、末行有效性，仍须在原始数据同步后逐条验证。下列关于其渲染时序的判断来自当前本地源码，不能替代源机器当时版本的核查。

## 差异与风险

### 1. 必须修正：UR7e 重渲染的物理时间与观测不同

G2 已完成任务的原始数值观测由 `G2CoreRecorder.record_pre_step` 保存，历史导出在第 0 帧恢复 `initial_state`，之后恢复 `states[t-1]`，不执行物理步。

UR7e 的 `rerender_embodiment_cameras.py:132-174` 则恢复 `states[t]`，然后执行 `sim.step(render=True)`。这里的 `states` 是标准 recorder 在 post-step 保存的状态，而 `obs` 是 pre-step 观测。因此按本地代码，图像比同一行观测晚一个控制步，再多一个物理步；15 Hz / 120 Hz 下约为 75 ms。

转换器 `convert_hdf5_to_lerobot.py:448` 仍把 `observation.img_state_delta` 全部填 0，不能作为时间已经对齐的证据。

**建议：**从 `initial_state + states[:-1]` 恢复每一行的 pre-action 场景。生产渲染不得调用推进物理的 `sim.step`；图像采样时另存 frame ID、时间戳、对应状态来源，以及真正的渲染验证结果。不能只平移旧视频一帧，因为旧视频还包含额外物理步的偏移，也没有原始第 0 帧。

### 2. 必须修正：隐含的末行删除及视频长度不一致

`convert_hdf5_to_lerobot.py:347-354` 对 state/action 无条件 `[:-1]`；有内嵌 RGB 时图像也 `[:-1]`，但旁路视频直接复制（529-545），保留全部帧。`dataset_config.py` 甚至把旁路视频多一帧描述为无害。

`lerobot_to_diffusion_policy_zarr.py:46-59` 解码到表格要求的帧数就停止，因此会掩盖多余帧，并把已有时间偏移带到 Zarr。本次 UR7e 18 条样本均存在这一个额外视频帧。

**建议：**先依据 HDF5 的实际 record hooks 确认 T 个有效动作，保留全部有效转移；若确有空闲尾部或任务截断，记录明确的 `[start, stop)` 区间、原因、源帧映射，并同时应用到所有模态。验证视频实际解码帧数必须等于 T，不能通过提前停止读取使检查通过。

### 3. 必须显式声明：动作、状态及末端标签不是同一种布局

| 项目 | G2 | UR7e |
| --- | --- | --- |
| 训练动作 | 16 维，右臂 7＋二值夹爪，再左臂 7＋二值夹爪 | 7 维，6 个机械臂关节＋`finger_joint` |
| 夹爪 | +1 开、-1 关 | 弧度，0 开、π/4 关；DP 派生为 0…1 |
| 原始训练标签来源 | `actions` | `joint_pos_target`，控制步末实际下发的关节目标 |
| 原始机器人状态 | 46 个关节的 pre-action `core` | `obs/robot_joint_pos`，全 articulation 12 关节，导出映射到 7 个受控关节 |
| 已有 observation.state | 60 维：46 关节＋双臂 14 维 TCP | 7 维受控关节；末端标签另存 |
| 末端表示 | 右后左 xyz＋xyzw，分别保存机器人根坐标系和世界系 | xyz＋旋转矩阵前两列（LeRobot eef_9d），在 UR `base` 坐标系；DP 再转换为前两行 |

相关代码：`g2_dataset_io.py`、`g2/recorders.py`、`ur7e/demo_recorders.py`、`add_ur7e_eef_9d.py`、`lerobot_to_diffusion_policy_zarr.py:38-43,127-139`。

**建议：**统一字段语义和单位，通过 robot profile 明确关节顺序、左右顺序、夹爪类型/单位、参考坐标系、TCP 链接及旋转编码。保留原始 command 与实际 applied target；不能拿下一步实测关节位置冒充 action。不同机器人的数据可由同一工具处理，但不能未经适配拼成同一种 policy action。6D 旋转和 DP 归一化只作为明确、可追溯的派生表示。

### 4. 必须修正：成功/结束标注与截断没有统一的证据链

G2 的 `read_core` 会验证成功报告，并执行任务相关物理检查。UR7e collectors 通常导出成功轨迹，主批处理也有质量过滤；但通用转换器对读到的每条轨迹都令末行 `next.reward=1`、`next.done=True`（442-447），并不自行核对该条 `success` 或截断后是否仍达成任务。

`truncate_demos_at_gripper_release.py` 按“最后一个达到最大闭合目标的步骤”截断，同时保留原始属性。它表达的是一种任务尾部策略，不是任意任务均适用的成功证明。

**建议：**区分 `episode_success`、`terminated`、`truncated`、`failure_reason` 和 `quality_valid`；成功来自任务判定/审计，不能从“这是最后一帧”推导。截断后重新检查最后保留的 post-action 状态。任务自然语言、稳定 task_id、原任务索引、资产方法和场景变体分别保存。现有脚本阶段信息可转成区间标签，但没有证据的人工动作阶段不得自动编造。

### 5. 需要统一：视频规范、标定、随机化与派生数据

G2 使用原生 head 640×400、两个 wrist 640×528；UR7e 使用经过标定的固定 D435 640×480，DP 另外缩放到 320×240。应保留这些相机配置，而不是为 UR7e 伪造两个 wrist 或强行统一分辨率。

UR7e 渲染按原 demo 后缀索引确定随机化（`rerender_embodiment_cameras.py:127-130`），转换器则按 HDF5 key 的遍历顺序编号（例如 `demo_10` 可排在 `demo_2` 前）。重新编号时如丢失原 demo ID，会改变随机化身份。

**建议：**每个视频保存 camera_id/role、分辨率、内参、畸变约定、安装位姿或逐帧外参、色彩空间、fps、编码设置与源状态映射。保留原始 demo ID、原随机化记录、资产哈希、源构建版本，不重新抽签。LeRobot 与 Zarr 都由同一规范化 episode 生成；resize/crop、关节映射、旋转转换、归一化、全局/episode 内时间戳必须有转换记录。

### 6. 需要统一：缓存、异常处理与运行配置

G2 有源文件哈希、原始文件摘要、episode 提交检查和不可混用的恢复配置；UR7e 主批处理也有 plan/checkpoint/构建身份及发布日志，这些机制可以复用，不必重写所有任务。

但通用转换器捕获单条转换异常后直接 continue（500-511），可能得到不完整数据集；旁路视频的存在及外观标记不能证明它对应当前原始状态、相机和时间语义。UR7e renderer 固定 `FPS=15`，且另建 Arena 参数时没有显式转发外层 `args.device`，不能仅凭启动命令推断实际 env 配置。

**建议：**统一阶段入口与 manifest：任务 profile → 采集 → 原始数据审计 → 渲染/现场 RGB 编码 → 导出 → 完整验收。每个阶段以原始哈希、schema、代码/资产/相机/随机化、时间映射决定缓存身份。失败明确登记并停止发布；如允许跳过，必须记录完整排除清单和预期/实际 episode 数。fps/device 从实际 env 或不可变采集元数据读取并断言。

## 推荐统一方案（待实现，不是当前已生效的 API）

### 原始层：不可变且可复现

保留旧 HDF5 与当时源码；新规范增加显式 pre-action 场景状态、post-action 场景状态、原始 command、实际 applied target、关节/末端/相机观测、frame_id 和 episode 内时间戳。保留完整 articulation 状态：抽屉和烤面包机也是有状态机构，不能只存机器人和物体根位姿。

episode manifest 包含：schema_version、episode_id、原始 demo_id/文件摘要、task/robot profile、instruction、seed、成功/结束原因、控制/物理 dt、动作来源与顺序、资产/代码/运行环境版本、相机标定与随机化来源。raw 是唯一事实来源；规范化文件和训练格式都是可再生成的派生物。

### 时间层：共用一个索引规则

老式 `initial_state + post-step states` 统一读为 `s_t = initial_state`（t=0）或 `states[t-1]`（t>0）。`a_t` 保持原始第 t 行；UR7e 明确选择 `joint_pos_target[t]` 作为相应训练标签。与 `core` / `obs` 数值逐行比对，发现 shape/时间不一致立即拒绝，绝不按最短长度自动截断。

### 图像层：两种获取方式，同一时间契约

1. 新采集优先支持同控制步的现场 pre-action RGB，这是已验证可用的严格对齐路线；代价是采集 I/O/存储较大。
2. 状态先采、离线重渲染也可以使用，但必须在不推进物理时间的情况下恢复 pre-action 几何和相机。GPU FK/状态恢复、渲染图变换刷新、相机采样需要分别验证。
3. 历史 G2 已通过的图像可经验收复用；不为“形式统一”重新生成全部视频。UR7e 需要根据同步后的源代码/原始数据确定重渲染清单。
4. 渲染验收同时比较 raw→物理 FK、raw→实际渲染几何/相机变换，并做首、中、末及接触阶段的画面检查。末端数值误差小不等于图像已经更新。首帧渲染预热不得改变用于采样的物理状态。

本次本机零步 CUDA 探针：Fabric 重挂载后末端数值误差约 6e-8 m，但画面仍停留在错误姿态；显式 USD 同步的另一路出现腕部黑图。因此这两种简单替换都**未通过**，没有接入生产。不能以它们的数值误差报告宣称严格对齐已经解决。若后续 GPU 无物理步重渲染尚未通过，应保留阻塞状态，或对新样本用现场 RGB；不应悄悄改用 CPU 物理或接受额外物理步。

### 标注/导出层：共同结构、机器人配置、可追溯转换

统一 episode/task/frame 标注和 schema，但保持 G2 16 维与 UR7e 7 维动作配置。明确 `joint_position`、`applied_joint_target`、`eef_pose` 的时刻/单位/坐标系。基础末端表示可统一为显式 frame 的 xyz+xyzw，现有 UR7e 9D、DP 10D 与夹爪归一化从基础层派生，并保留原表示以校验兼容性。

所有训练行必须满足：`N_action = N_obs = N_image = N_annotation = T`；相机 PTS 和 episode 内时间戳一致；视频完整解码；原始数值/标签可回溯；成功与有效性不混淆。没有证明对齐的产物不能标为统一数据集完成。

## 后续实施顺序

1. 同步 UR7e 原始 HDF5、逐 demo 随机化/外观记录、生成它们的脚本与配置、USD/URDF 及依赖资产版本、采集/导出 manifest。保留旧成品。
2. 固定任务清单：G2 2 个；UR7e 按仓库 12 个 case 与源机器实际库存核对。没有数据的 case 列为缺失，不混入 smoke/资格测试轨迹。
3. 实现共用时间适配器、机器人/任务 profile、原始数据/视频契约及统一验收命令，再接入现有 collectors 和两个 exporter。旧 Agibot 不改。
4. 先做 G2 两个任务与 UR7e 抽屉/按压/旋钮各代表 case 的小样本端到端验证；各资产变体分别核验。通过后才批量重渲染到新目录。
5. 同一原始 episode 分别走规范化→LeRobot、规范化→Zarr，逐行验证动作、状态、末端、夹爪、图像时间和任务标注。
6. 全量完成后输出旧→新 episode 映射、重用/重渲染/拒绝清单、质量报告，再讨论切换数据入口。本次不执行同步、迁移、正式重渲染或发布。

## 当前工作树边界

本次仅新增本审核文档与 `outputs/unified_collection_20260926` 下的只读审计结果/诊断探针。早期统一接口草稿已移至该实验目录的 `draft_not_integrated`，未安装、未接入 collectors。此前工作台清理的既有修改保留；没有提交、推送或改动原始/正式成品数据。
