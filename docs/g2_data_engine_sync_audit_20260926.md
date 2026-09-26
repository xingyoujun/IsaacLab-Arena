# G2 与最新 data_engine 合并审核

日期：2026-09-26。结论：代码已本地合并；G2 可作为独立机器人和独立场景接入相同资产体系，但尚未完成可移植资产封装、公共规划入口及统一训练数据链路。没有推送 Git、上传 HF 或启动任务批量采集。

## 本次同步

- 工作区：`/home/ubuntu/code/IsaacLab-Arena_g2-integration`，分支 `xingyoujun/feature/g2-data-engine`。原始工作区保持原状。
- 来源：`xingyoujun/IsaacLab-Arena` 的 `chuanruiz/data_engine`，最新提交 `ccefc86a4b0aed15bbc431959d3932fa50e1a8f1`（pine wm init）。
- 本地工作先保存为带 DCO 的 `c893daa3`；带 DCO 的合并提交为 `6945c3d7`。新增远端别名 `fork`，原 `origin` 保持不变。
- 合入 53 个文件，9361 行增加、14 行删除；无文本冲突。本次远端未改 Docker、工作流、pre-commit 配置或子模块。
- HF 固定版本 `d0bfcf1fcda19753569e222c834386509e40e1b5`；manifest SHA-256 与 `tools/usdcraft_scene/release.json` 一致。下载到忽略目录 `local_assets/USDCraft-Scene/`。
- 合并点相对 data_engine 仍有 72 个文件差异、9208 行增加和 11 行删除，包含已有 G2 两任务、工作台清理和兼容改动。反向整合不能只挑本轮 34 个文件，也不宜未经审核直接发布整个分支。

## 先纠正任务归属

新代码的 `docs/task_families.md` 明确区分 RR real2sim 和 Pine WM。此前本机六套 UR7e 成品属于 RR；其视频多一帧的抽样结论不能外推为 Pine WM 数据已经存在相同问题。

本轮统一目标应按 G2 的 `stack_bowls`、`peg_into_sleeve` 和 Pine WM UR7e 已实现任务审核，并为工作台清理沿用相同协议。旧 Agibot 排除；RR 保留独立归属，不能默认为 Pine WM 的替代数据。源机器原始数据仍待后续同步。

Pine WM 有 first20 的 20 项实现；200 项目录是规划清单，不是 200 项已完成采集。本提交文档曾写 preview 已开启，但 `review_layouts.json` 实际 `launch_authorized: false`；`review_workflow.validate_run()` 只接受单任务 preview，拒绝稳定性和采集阶段。因此本机当前配置不能直接启动批量采集，代码与文档存在状态不一致。

## 主要差异及影响

| 项目 | 当前 G2 | 最新 Pine WM | 接入要求 |
|---|---|---|---|
| 机器人 | 固定底座 G2，双臂及两夹爪 | UR7e + Robotiq，单臂，专用安装件 | 独立 embodiment，保留各自关节顺序和动作定义 |
| 规划 | cuRobo MotionGen；工作台另有 native cuMotion 适配 | 注册的 `pine_wm_ur7e` cuMotion 配置 | G2 注册左右臂配置；不可借用 UR7e/Agibot 的 TCP、碰撞球或限位 |
| 控制/状态 | 历史导出 state 60、action 16，双夹爪开闭符号 | 7 维绝对关节目标，夹爪连续弧度 | 统一字段语义和元数据，不强制等维度 |
| 三路相机 | head 640×400；左右 wrist 640×528 | D435 与两个 wrist 均 640×480；另有诊断 scene_cam | 保存原始分辨率、内外参、附着坐标系及相机角色，训练缩放另记变换 |
| 坐标 | 工作台表面 z=0，机器人底部在负高度 | 表面 z=0.74，机器人位置 (0,-0.425,0.75) | 记录 world/base/tool/camera 变换，不能直接复用抓取坐标 |
| 资产 | GenieSim、Arena 资产、Git 小型 USD 和本机规划文件 | HF manifest 稳定 ID + 固定 release；任务语义留 Git | G2 补依赖闭包、版本和来源 |
| 成功验收 | 工作台三物归框、释放、静稳与轨迹审计 | 每任务目标、接触/碰撞审计、成功和失败分开 | 共用验收输出结构，保留任务专属判据 |

### 图像对齐没有随 merge 自动统一

1. Pine WM first20 的状态重渲染改为 `initial_state + states[:-1]`，对应执行前观测；用 1 微秒物理步刷新图像，再从物理后端验漂移。它是有界近似，不是数学上的零漂移重放。代码还保留 `visual_sync_review_required: true`，数值通过不代替画面审核。
2. 该分支只在 `args.env == "pine_wm_first20"` 时启用。G2 工作台的 `g2_workcell/rerender_states.py` 仍读取当前 post-step state，再推进常规物理步。之前成功视频可作任务展示，不能据此声称训练图像与 pre-step 观测严格对齐。
3. 最新 `live_preview.py` 在 `env.step()` 之后抓取实际传感器帧；manifest 明确标记 `live_sensor_post_step_preview`。这条展示链路与 pre-step 训练重渲染链路必须分开管理。
4. 公共 LeRobot 转换器仍无条件删除 state/action 末行，旁路视频仍另行复制。新 Pine WM recorder 对 pre-step 对齐作了断言，但不能由此推断导出链路已经统一；需原始文件核对有效 T 步后同步裁剪所有模态。
5. 三者仍包含 CPU 操作：图像 `.cpu().numpy()`、libx264 编码及文件 I/O。RTX/CUDA 渲染不等于整个流水线均在 CUDA。原生 cuMotion 的 GPU 执行不能只凭接口名称确认。

建议抽取按机器人配置的公共重渲染器：显式选择状态对应时刻、物理微步及漂移阈值，验三路运动/非空图像、帧 ID 和时间戳，再验证视频实际解码帧数 = 观测行数 = 有效动作数。先用各机器人少量原始回合验证，不直接全量重渲染。

### G2 资产接入的具体缺口

- `g2.py` 仍从 `GENIESIM_ASSETS_DIR/robot/G2_omnipicker/robot_fix.usda` 读取机器人。
- 桌子来自 GenieSim `background/common/table/benchmark_table_000/Aligned.usda`；碰撞包装器写入用户缓存，并引用传入源路径。直接复制缓存会携带机器路径，而且现有缓存按 basename 命中，不跟踪源内容变更；打包需相对引用或按资产哈希生成。
- 电钻、齿轮和 bin 分别来自 Arena object library 的 `cordless_drill_ycb_robolab`、`small_gear`、`bin_b04_vomp_robolab`，不在本次 HF 清单内。房间、铝块的轻量 USD 在 Git；场景缩放、颜色、质量和碰撞设置属于运行时配置，也要固定版本。
- 采集器默认依赖 `/datasets/agibot_dataset_v1_raw/assets/g2/robot.yaml` 与 `/datasets/g2_workcell_assets/g2_tcp_aligned.urdf`。这些描述及其网格依赖未随 Git/HF 发布，另一台机器不能只 clone 后直接运行。
- 当前 HF `prepare()` 专为 Pine WM 来源目录编写，不能直接拿来封装 G2。`upload()` 要求保留远端全部文件，并用 parent commit 防并发覆盖；接入时必须合并完整 manifest、保留现有稳定 ID，不得生成仅含 G2 的替换 manifest。
- 来源/许可说明应随 GenieSim 和 Arena 对象保留；HF 私有存储不会自动变更上游授权。这里只核对源码中的来源声明，未完成逐资产再分发许可审核。

建议布局（待实现，不是已发布条目）：

```text
embodiments/g2/robot/          # USD、网格、材质，保持相对引用
embodiments/g2/calibration/    # 相机来源快照；Git 中明确唯一权威配置
embodiments/g2/planning/       # 外部规划描述及其依赖
scenes/g2_clean_workcell/      # 场景描述、资产绑定、布局快照
assets/<stable_object_name>/  # 桌子、铝块、电钻、齿轮、bin，可跨场景复用
```

Git 保留机器人控制、任务逻辑、采集/导出代码、规划适配、测试和 release 锁。HF 存储模型载荷、来源和文件哈希。派生配置若两处保存必须有一致性校验，不能形成两套独立修改的场景/标定。USD 跨目录、移到新根目录后须验证所有依赖闭合。

## 建议实施次序

1. 整理 G2 资产和规划依赖清单，新增稳定 ID 解析并保留显式旧路径兼容；在临时新根目录验证机器人、工作台、三物三框可以加载。
2. 将 G2 左右臂规划配置接入公共 registry，保留可选 cuRobo/native cuMotion 后端，验证 TCP/FK、限位、自碰撞和持物避障。
3. 抽取统一 raw metadata、成功/失败标注、相机映射和有界重渲染协议；保存 Git commit、HF revision、配置哈希、随机种子、有效帧区间和漂移报告。
4. 对 G2 两个既有任务、工作台清理及 Pine WM 代表任务分别回归；含机器人和物体运动、接触、腕部图像和末帧检查。原始数据到位后才判定哪些可直接转换、哪些必须重渲染。
5. 反向提交前审核 72 文件差异：尤其公共 keyboard/registry、包数据，以及此前用于环境注册的 CAP USB-C 延迟导入兼容改动；将通用兼容修复与机器人/场景功能清楚说明。先完成本地可复现，再另行发布代码与 HF 新 release。

## 本次验证与边界

- 合并前本地工作：全量 pre-commit 通过。
- 合并后：G2 workcell、HF 管理器及 Pine WM 几何共 32 个相关单元测试通过。
- Isaac Sim 中额外运行 G2 任务契约测试：2 项通过（29.73 秒），验证既有任务成功/终止约定；合计本轮 34 项测试通过。
- HF 全部 49 个清单文件共 20,245,055 bytes 校验通过；21 个 USD 入口依赖闭包通过，仅保留明确的 Isaac Sim `OmniPBR.mdl` 运行时依赖。
- 远端两个 CSV 使用 CRLF，`git diff --check` 报行尾空白；本次保留远端内容，未混入整库格式重写。这不属于无冲突合并的运行时失败，但发布前需处理格式检查。
- 日志与 manifest 快照在 `outputs/data_engine_merge_20260926/`。本次没有重跑完整 G2 三物清理、Pine WM 全任务或完整三阶段测试，不能把代码合并/单元通过当成全量采集验收。
