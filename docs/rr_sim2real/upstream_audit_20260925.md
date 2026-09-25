# RR sim2real 本地整理与 upstream main 升级评估

后续状态：用户已批准本地合并但保持 Sim 6.0.1；实施记录见
[迁移报告](migration_20260925.md)。以下保留为合并前的只读审计快照。

检查日期：2026-09-25。结论：保留当前可复现实验基线；值得在隔离分支和独立
Python 环境评估新 main，但不建议在当前工作区直接 pull/merge 或更新共享 `.venv`。
本次没有执行实际合并、提交、推送、安装依赖或 GPU 测试。

## 1. 精确版本与检查范围

- 当前分支：`chuanruiz/rr_sim2real`。
- 本地 HEAD 与实时查询的 fork 同名分支都是 `18b52440c7a0f323055858bc80ce07837641557e`。
- 实时查询的 upstream main：`aa36f191d89b1ea65fbba62b3350d6fc7d0e9b9f`
  （提交时间 2026-09-24 UTC，Isolate Isaac CAP tests from core suite）。
- 共同祖先：`af3f24b054879a3886e80447772cd27e4bf208f1`。
- 上游独有 64 个提交；本地独有 37 个提交，包含此前继承的 Agibot 开发，
  不应将这 37 个提交全部算作本次 UR7e 工作。
- 上游自共同祖先起变更 448 个文件，约 +50,541/-4,824 行，含大量文档、
  CAP 示例及配置；这些数字不是 UR7e 所需改动量。
- **本地 HEAD 的 `pyproject.toml` 已声明 `version = "0.3.0"`**。
  这不是从纯 v0.2 升级到 v0.3，而是同步 0.3 开发线后续提交。
  Typed Experiments、HTML 报告、variations、agentic environment generation 等
  框架本地已经存在，不能把整份 v0.3 release notes 当成此次新增收益。
- `git ls-remote` 本次未返回 `v0.3*` 标签；评估对象是上述确切 main SHA，
  不是一个已核验的 v0.3 release tag。

上游代码只下载到 `/tmp/rr_arena_v03_audit_PLoERD/repo.git`，不更新共享仓库
的 remote-tracking refs，不碰兄弟 checkout 的工作区/分支。该临时目录不是备份。

## 2. 本地未提交内容如何整理

检查开始时共 57 项：17 个已跟踪文件修改、3 个删除、37 个未跟踪文件；
暂存区为空。已跟踪 diff 为 +263/-259 行；37 个新文件合计 168,365 bytes。
本审计文档是检查后新增的，因此后续 status 数量会增加。

建议按以下主题审查、组织后续提交，不按一次性的运行日期散落：

| 主题 | 主要文件/目录 | 必须保留的约束 |
| --- | --- | --- |
| 资产命名与四方法适配 | `asset_naming.md`、`assets.json`、`ur7e_rr_sim2real_environment.py`、各 baseline env、`asset_overlays/` | 8 个资产、3 类任务；只缩放 Articraft；保持旧 HDF5 articulation keys |
| 黑色夹爪 | `embodiments/ur7e/appearance.py`、`ur7e.py`、re-renderer | 指尖直接材质绑定；不改碰撞/摩擦；缓存标识 `black_fingertips_direct` |
| cuMotion 与数据管线 | drawer/press drivers、`prepare_rr_sim2real_all.*`、knob driver/pipeline | 200 demos/case、随机背景与干扰项、抽屉拉开即结束 |
| 旋钮任务 | `ur7e_turn_toaster_knob_environment.py`、boxx overlay、qualification/docs | 两个盒子；原方向；双向旋转绝对值超过 2°；reset 清位置和速度 |
| DP 训练、评估、传输 | `eval_all_dp.py`、`train_toaster_knob_dp.sh`、HF scripts、DP client audit 日志 | 训练与评估分环境；每 case 独立统计；双 worker 独立端口/录制文件 |
| 交接文档 | `AGENTS.md`、RR README、任务说明 | 区分历史资格检查与当前完成状态，不覆盖历史结果 |

三个 deleted overlay 为旧名称 `drawer_rr_arena.usda`、
`drawer_rr_gpt56_arena.usda`、`toast_rr_arena.usda`；新 canonical overlay 文件
当前是 untracked。发布时必须一起检查，不能只提交删除或遗漏新路径。

没有在这 37 个新文件中发现大数据/权重文件；最大的脚本约 27 KiB。
这不是全面敏感信息扫描，未来提交仍需检查 staged 内容，不能直接 `git add .`。
当前 `git diff --check` 通过；未运行完整 lint、pytest 或新版模拟器验证。

### 当前数据与评估事实

本次读取正式数据根目录下 12 份 `meta/info.json`，每份均为 200 episodes，
相邻 DP Zarr 均存在；各 LeRobot `rr_sim2real_build.json` 声明
`randomize=true`、夹爪 `black_fingertips_direct`。这是元数据核验，不是重新逐视频验证。

2026-09-16 的完整 DP 评估 manifest 为 `completed`、240 episodes：

| 方法 | open_drawer | press_toaster | turn_toaster_knob |
| --- | --- | --- | --- |
| USDcraft | 12/20 | 16/20 | 12/20 |
| Articraft | 20/20 | 8/20 | 15/20 |
| miniworkflow GPTSOL | 20/20 | 19/20 | 18/20 |
| miniworkflow Astra | 17/20 | 19/20 | 18/20 |

结果在 `outputs/dp_eval12_20260916_retry1/`；完整协议见 [eval_all_dp.md](eval_all_dp.md)。
这是随机物体位置、标准评估背景下的结果，不能声称评估也启用了训练背景随机化。

RR README、asset naming 文档及 assets.json 的部分状态仍停在 9 月 11 日：
“待黑色重渲染”“尚无 baseline 训练数据”“尚未批量采集”等已过时。
`assets.json` 有四个 dataset 字段仍为 null；实际对应正式数据已存在。
旧 200 epoch / batch 64 示例不代表后续 80 epoch / batch 128 的训练交接协议。
下一次发布前应统一修正文档状态，保留旧实验的日期/协议，不直接抹掉历史记录。

### 外部 diffusion_policy 仓库也需要独立保存

只读检查发现其已修改 `train_diffusion_unet_image_workspace.py`：checkpoint
改为按已完成 epoch 间隔保存，并强制保存末轮。这是 epoch 79 交接的重要补丁，
不是“仅新增四个文件、原文件完全未改”。另外训练脚本、server、dataset adapter、
task YAML 和交接 README 仍为 untracked。

`ckpts/` 也出现在该仓库 untracked 列表，**不得随代码提交**。Arena 中的 DP
overlay 与实际 DP 文件并非字节一致（已检查的差异包含版权头、格式、import/类型注解）；
不能直接拿格式化 overlay 满足训练脚本的源码 SHA256 检查，也不能仅凭哈希不同
就认定算法改变。发布时应附实际补丁及明确的源码/配置哈希。

## 3. 新 main 对我们的实际价值

| 更新 | 对 RR 的意义 | 限制与优先级 |
| --- | --- | --- |
| [rollout timing #1198](https://github.com/isaac-sim/IsaacLab-Arena/commit/dd91b2006b6b3386a49f61272aef73faeed2d430) | 拆分环境 step、推理、视频等耗时，判断双 worker 是否真有收益 | 高；可评估局部移植，不能假定自动加速 |
| [evaluation trajectory recording #1254](https://github.com/isaac-sim/IsaacLab-Arena/commit/d8577ffc38e6c315148297ca0c225b510e1fd878) | 保存逐步状态/动作，方便分析抓空、IK hold、按压失败 | 高；需接回 UR7e 自定义录制契约，不直接替换旧数据格式 |
| [placement replay #1291](https://github.com/isaac-sim/IsaacLab-Arena/commit/61236e66dcab320125c632f1aee0c1add9dc31ff) | 固定测试布局，改善四方法对比的配对性 | 高；只恢复 root poses、不恢复机器人关节；不同资产 key 和自定义 stack reset 需适配 |
| TaskTerminationCfg + temporal predicates | 统一成功、进度和持续条件，适合将来组合任务/稳定成功判断 | 长期有益；本轮必须保留旧成功口径 |
| Isaac Sim 6.1 + 恢复多次 rebuild 的 GPU/Fabric | 上游移除了多次构建时强制 CPU 的 workaround | 中；我们当前每进程仅一个 run/rebuild，本就未触发该降速分支 |
| gripper interface、spawn physics addon | 将夹爪状态/释放判据、任务级物理配置模块化 | 中；不自动修复 Robotiq 指尖材质，也不自动适配 UR7e |
| distractor-disappear variation | 可增加测试时干扰物消失的鲁棒性消融 | 可选；与现有离线背景随机重渲染不是同一件事 |
| CAP 齿轮、USB-C、线缆、柔性体、Newton DROID | 可作将来接触丰富/插入任务参考 | 对当前抽屉/按压/旋钮收益较小，不应因此重做现有实验 |

v0.3 的官方概览见 [release notes](https://github.com/isaac-sim/IsaacLab-Arena/blob/aa36f191d89b1ea65fbba62b3350d6fc7d0e9b9f/docs/pages/references/release_notes.rst)。
上述价值判断是依据实际 diff 对 RR 工作流作出的判断，不是上游宣称的 UR7e 兼容性保证。

## 4. 已确认的迁移风险

1. **采集导入会失效**：新 main 删除
   `isaaclab_arena/utils/isaaclab_utils/recorders.py`；
   `embodiments/ur7e/demo_recorders.py` 仍从这里导入两类 recorder。
   需要保留兼容实现或迁移，并检查 `joint_pos_target`、action 和 camera 时序。
2. **旧 IdleTask 接口不兼容**：`ur7e_workcell_environment.py` 返回旧 termination
   configclass；新 builder 明确 assert 返回 `TaskTerminationCfg`。
3. **成功判定可能变口径**：三个任务族都复用 `OpenDoorTask`。上游增加
   “相对 reset openness 移动超过 joint range 的 5%”再进入 is_open 的序列。
   我们的旋钮 `is_open` 是绝对角度超过 2°，reset 是绝对零弧度，而非通用
   normalized openness。不能假定这两个前提在所有资产上等价；必须显式保留
   当前阈值、双向语义、reset 行为及成功帧时序。
4. **运行栈升级不是只合代码**：本机 installed metadata 为 Isaac Sim 6.0.1.0、
   IsaacLab 16.4.0、torch 2.11.0+cu128、numpy 2.3.1（metadata 不代替运行时探测）。
   新 main 的 uv override 指定 Isaac Sim 6.1.0.0，mujoco-usd-converter
   从 0.2.0 改到 0.5.0，新增 pytetwild（本机未安装）；Plotly 已安装。
   IsaacLab 与 Isaac-GR00T 的 gitlink SHA 在两边相同，但依赖约束并不相同。
   当前 `.venv` 指向兄弟主 checkout，严禁就地 `uv sync`/pip 升级。
5. **文本冲突**：对已提交 HEAD 做只读三路 merge-tree 预检得到 5 个冲突文件、
   7 个 conflict hunks：`arena_env_builder.py`、`isaaclab_arena_environment.py`、
   `tasks/predicates/spatial.py`、`tasks/task_library.py`、`terms/events.py`。
   两边已提交改动有 13 个文件重叠；当前 dirty tracked 路径与上游变化仅
   `AGENTS.md` 重叠，37 个 untracked 路径无上游新增同名碰撞。
   此检查不等价于对完整 dirty tree 的真实 merge，更不代表运行兼容。

## 5. 建议执行顺序（尚未执行）

1. 先整理并保存现有 RR 代码基线：按主题审查提交，更新过时交接状态，保存
   外部 DP patch/config、资产/数据 build identity 和 240 集旧结果的关联。
   只包含小代码/文档，不带权重、数据、视频；是否提交/推送由用户确认。
2. 在独立 clone 或隔离分支建立升级试验，锁定上游 SHA；使用独立 runtime，
   保留原 checkout、资产、数据和共享环境。不要直接把移动中的 main 当基线。
3. 修上述 API/录制兼容点，并显式选择原 PhysX 后端；不同时更换成功标准、
   资产几何、D435、TCP、action/state 格式或训练配置。
4. 先 12 环境构建/reset/黑色指尖与摄像头检查，再每种任务的短 cuMotion
   states-only→重渲染→LeRobot/Zarr smoke；确认标签、帧数及旧 HDF5 可重放。
5. 同一批 checkpoint、同一组固定姿态、同一 timeout 下做新旧 DP 对照。
   验证通过再考虑迁移主开发线；新版结果单独成组，不能覆盖 9 月 16 日结果。

短期只想解释现有失败时，不必整体升级。优先在旧基线上补诊断轨迹/耗时，
审查 USDcraft 抽屉、旋钮失败视频；新 main 本身不构成提高策略成功率的证据。
