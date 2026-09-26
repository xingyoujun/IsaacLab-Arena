# pine_wm 首批 20 项任务资格测试

资产源：`/home/ubuntu/playground/pine_wm_first20_physx_20260925`，来自用户的同名 zip。
19 个资产的 SHA256 已与原包逐项校验；原始 USD、网格和纹理保持原样。
资产静态审计在 `/home/ubuntu/playground/experiments/pine_wm/first20/asset_audit.json`。

**当前是调试与资格测试，不是已完成的训练数据集。** 各次尝试单独保存，失败不能被删除后算成功率。
旧 RR 抽屉成功记录不适用于本包的 280 mm 新抽屉；它使用横向拉杆，而非球形把手。

环境入口 `pine_wm_first20`，通过 `task_id` 选择任务。任务定义和资产绑定在
`tools/pine_wm/first20/tasks.json`，实例、固定夹具和资产适配在
`isaaclab_arena_environments/pine_wm_first20_environment.py`。

## 随机布局与验收

默认单物体中心采样区域为世界坐标 x ∈ [−0.30, 0.30] m、y ∈ [−0.12, 0.25] m；
桌面 z=0.74 m。可以通过 `--workspace XMIN XMAX YMIN YMAX` 修改。
物体 yaw 覆盖 ±180°。新抽屉开口 yaw 先提案 ±60°。
写入随机位姿后先刷新运动学并检查机器人/桌沿净空，再开始静稳；静稳过程也检查物理接触。
复合任务使用任务相关布局，不能把多个物体独立采样到同一位置。
箱体、目标点、抽屉全行程、夹爪与相机需要共同通过可达性/碰撞验证。
抽屉采样检查完整 150 mm 行程的桌沿净空；复合布局记录重新采样次数。
T031/T038 默认交替测试 3 层与 5 层，可用 `--stack-count 3` 或 `5` 固定。
T145 每回合从 6 件中采样 N=2..5，N 写入语言指令与数据属性。
T031/T143 的源积木次序、T142 的源形状次序独立打乱，避免固定站位泄漏答案。
T142 三种待分类形状使用相同材质，消除原资产颜色与类别的固定绑定；原 USD 不改写。
T143 排列中心间距由草案的 80 mm 调整为 120 mm，给完整夹爪留出取放空间。

每条结果保存原始提案、静稳后的初态、任务目标、失败阶段、实际末态和成功指标。
拒绝不可达或不安全提案不等于任务成功，也不允许悄悄缩小范围后仍宣称覆盖原区域。
验收目标为最终同一版本至少 20 个随机回合、至少 95% 任务成功，另检查工作区覆盖、
无禁止的执行接触、数据重放与三路相机一致性。开发过程中的单回合成功只用于定位下一步。

## 碰撞与物理

- 相机壳、打印支架、垫片增加真实 PhysX 碰撞形状，支架孔洞采用保守凸包。
- cuMotion 自碰撞、机器人/相机球体与物体逐碰撞体 OBB、夹爪真实碰撞体 OBB共同检查。
- 检查时间参数化轨迹及其插值；执行后再次检查实测状态。
- 携带物体跟随夹爪预测运动，也检查携带物体对桌面和其他资产的干涉。
  携带物体包围球加入 cuMotion；独立细几何检查仍继续执行。
  圆柱/球使用圆形支承函数，网格的最低点使用真实顶点，避免外接方盒误报穿桌。
  圆柱/球对托盘等 OBB 的支承间隔也使用实际圆形投影。
  `audit_tool_spheres.py` 对打印支架的凸包表面采样，补齐 wrist_a 支架与相机/套环球之间的覆盖缺口。
- GPU PhysX 接触力审计补充几何检查。只有指定操作物体与内侧夹指的预期接触允许发生；
  外侧夹指、手掌、机械臂和腕部相机撞到盒沿不能算成功。
- 深盒落放使用盒沿上方释放，避免夹爪伸到狭窄内腔；具体稳定性仍要逐任务验证。
- 圆柱/球体在此场景中增加有限滚动阻力（angular damping=2.0，linear damping=0.1），
  处理长回合中理想无滚动阻力物体滚出有槽桌面的问题；没有冻结待抓物体。
- 夹具使用固定/运动学基座；不通过瞬移、吸附、关闭物体碰撞来伪造完成。

## 运行

本工作树使用原有 native 环境，不安装或更新共享依赖；只开一个 GPU 仿真进程。

```bash
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y
export PYTHONPATH=/home/ubuntu/code/IsaacLab-Arena-tasks:/home/ubuntu/code/IsaacLab-Arena/submodules/IsaacLab/source/isaaclab
.venv/bin/python tools/pine_wm/first20/qualify.py \
  --task T001 --trials 20 --seed 100 \
  --output /home/ubuntu/playground/experiments/pine_wm/first20/qualification/T001
```

顺序跑全部任务（输出目录应为新的轮次）：

```bash
python3 tools/pine_wm/first20/run_batch.py --trials 20 --seed 100 --require-frozen \
  --workspace -0.30 0.30 -0.12 0.25 \
  --output /home/ubuntu/playground/experiments/pine_wm/first20/qualification
python3 tools/pine_wm/first20/summarize.py \
  /home/ubuntu/playground/experiments/pine_wm/first20/qualification
python3 tools/pine_wm/first20/qualification_report.py \
  /home/ubuntu/playground/experiments/pine_wm/first20/qualification
```

每任务写 `runtime.log`、`results.json`、`environment.json` 和成功回合 `demos.hdf5`。
有动作记录的失败回合另存 `demos_failed.hdf5`，仅供诊断，不能并入训练成功集。
批处理另写启动时的 `source_hashes.json`；调试期间代码可能继续修改，正式验收应冻结版本。
结果中的 `planning_rejections` 保存未执行的候选路径拒绝，不能与实际碰撞混淆。
`min_clearance_m` 包括拒绝候选；`min_executed_arm_camera_sphere_clearance_m` 只统计执行状态的球体净空。
两者都不代表整个夹爪所有表面的全局最小距离，夹爪另由 OBB 和物理接触审计把关。
记录从随机布局静稳后开始，排除 reset 瞬移与物体下落静稳过程。
动作维持 7 维绝对关节目标与 15 Hz；相机通过状态重渲染生成。
当前验证的 first20 重放使用 1 微秒物理步刷新渲染端，逐帧从物理后端读取漂移；
不能标为数学上的精确重放。每个视频回合另写 `*_state_replay.json`，记录关节/根位置误差和步长。
T001 的四路 461 帧、T044 的四路 789 帧回放已通过状态漂移、画面变化和人工抽帧检查，15 Hz；
三路任务相机为 640×480，第三视角为 1280×720。其余任务仍需逐项复核。
第一版仅调用 forward 的回放曾出现后端状态正确而画面停留在默认姿态的问题，已经弃用。
新缓存必须带 `validation_passed: true` 才可复用，失败版本不会被缓存命中。
图像对应动作前观测：第 0 帧取 initial_state，随后取前一控制步的 post-step state。
漂移上限分别为机械臂关节 0.001 rad、所有夹爪关节 0.005 rad、物体位置 0.5 mm、
物体姿态 0.005 rad、抽屉/按钮关节坐标 0.5 mm；实际最大值逐回合保存。
初次成功 T001 回放的实测位置漂移约 0.163 mm，机械臂最大关节漂移约 0.000756 rad。
固定相机可能被机械臂遮挡；腕部相机提供互补视角，不要求每路相机始终看清整个任务。

```bash
.venv/bin/python isaaclab_arena_cumotion/scripts/rerender_embodiment_cameras.py \
  --env pine_wm_first20 --env-config OUTPUT/environment.json \
  --hdf5 OUTPUT/demos.hdf5 --headless \
  --streams realsense_d435_rgb wrist_a_rgb wrist_b_rgb scene_cam_rgb
```

不要用默认 T001 场景重放其他任务。新录制文件包含 `env_cfg`，重渲染会自动读取；
旧文件必须传对应的 `environment.json`。不同 task_id 的文件分开重渲染。
`--enable_cameras` 可在测试期间输出每回合四视图 PNG，但会增加 GPU 开销。

## 验证范围

以上是仿真资格测试。自碰撞采用 cuMotion 球体与额外腕部相机检查；
源机器人关闭的 PhysX 自碰撞开关未改变，因此不能把本报告解释成实机安全认证。
最终验收还需要每任务的随机回合统计与三路相机状态重放证据，目前尚未完成。

## 调试策略记录

- 抽屉横杆使用向外倾斜 30° 等候选姿态，接近/释放时先缩小夹爪开口；箱体碰撞不豁免。
- 放置位置根据抬升后、运输后的实际夹持偏移修正，避免把“TCP 到位”当成“物体到位”。
- 瓶子扶正使用瓶颈抓取。原规格没有指定落点；若采样落点的立瓶姿态不可达，尝试桌面另一处可达空位。
  `requested_target` 与最终 `target` 均保存，不改变瓶子初始位置的采样范围。薄牌采用越过竖直的物理翻转和重力落位，再重新顶抓精确放入目标框；最终检查另一面朝上、目标框内与静稳。
- 薄牌翻转释放后不追加已知会自碰撞的倒置向上直线撤离；待静稳后重新顶抓。

## 失败布局复现与视频批处理

`qualify.py --replay-results PATH/results.json --replay-trial 2 --trials 1` 可复现某条提案，
无需先执行同种子的前两次回合。该输出带 `layout_replay`，是诊断回合，不是新的随机覆盖证据。

```bash
python3 tools/pine_wm/first20/render_batch.py QUALIFICATION_DIR
```

默认每任务渲染首条成功演示的四路相机，并写 `camera_replay_summary.json`；
`--all-demos` 渲染所有成功演示。批处理只生成数值/画面变化检查通过的视频，
不能代替对任务可见性与场景内容的逐项复核。失败仍留 `.part.mp4` 与漂移报告供诊断。

## Superseding workflow: 2026-09-26 user layout review

The user stopped further attempts. All old broad-workspace results are historical,
not acceptance of the revised layout. See [all-task layout review](pine_wm_review/README.md)
and `tools/pine_wm/first20/review_layouts.json`.

Execution is currently locked (`launch_authorized: false`). `qualify.py` and
`run_batch.py` reject stability/collection launches during this review. Once the
user requests a specific preview, enable only the reviewed preview workflow;
`run_batch.py` permits a single task, defaults to one trial, and `qualify.py`
stops at the first success even when a larger attempt cap was requested. No
automatic progression from preview to stability or data collection is allowed.

Drafts use fixed role-dependent positions near the table center. The drawer pose
is based on round08 T044 demo0 (not a claim that every historical T044 layout was
approved). Original USD geometry and camera calibration are unchanged. Tasks
T017/T038/T044/T142/T145 require container-size/layout revision before preview.
Randomized position/orientation/size ranges will be designed and approved after
the visual layout; broad legacy sampling is not used by the default runner.

### Single-success previews authorized later on 2026-09-26

The user explicitly authorized adjusting all twenty tasks and publishing one
successful video per task. `launch_authorized` is now true for preview only.
Stability and collection remain disabled. Each simulator process has a one-trial
cap; the queue does not automatically retry failures. Failed tasks are diagnosed
and revised separately. `review_v2` is external preview output, not a qualified
training dataset. Container instance scales are persisted in environment.json;
source USD files and calibrated cameras are unchanged.

Four camera streams are now captured from actual simulation steps by
`live_preview.py`. These are 15 Hz post-step visualization videos, not replayed
or training-aligned observation videos. A success and nonempty/moving image
checks are required before publication. Camera execution remains RTX on cuda:0;
this change does not claim all motion solving has migrated to CUDA.
