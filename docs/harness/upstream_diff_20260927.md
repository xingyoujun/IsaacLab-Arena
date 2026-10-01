# 相对 Isaac Lab-Arena main 的改动与源码归属

对照日期：2026-09-27。本轮只 fetch 上游引用，没有合并上游或修改主克隆工作树。
基线是官方 `origin/main`：`aa36f191d89b1ea65fbba62b3350d6fc7d0e9b9f`
（2026-09-23，Isolate Isaac CAP tests from core suite）。当前分支 HEAD 为
`09a786fb43241fc1d3a402f2b07cbcb935e94ee0`，包含该上游基线；以下也包含本工作树尚未提交的改动。

本地 `main` 不是官方最新 main，不用它作为上游基线。当前分支长期积累了 RR、Agibot、G2
开发，因此“相对 main 的全部差异”大于这次 Harness 目录整理。

## 已落地的主体模块

| 内容 | 当前源码位置 | 相对上游新增的能力 |
|---|---|---|
| Harness 核心 | `data_engine/harness/`、`cli/harness.py` | 配置解析、运行身份、单次 preview、GPU lease、超时/取消、不可变产物、独立 audit、资格状态 |
| 公共 cuMotion | `data_engine/motion/cumotion/` | 同一 planner、执行器、机器人描述转换与抓取坐标工具，供 G2/Pine 共用 |
| 机器人规划适配 | `data_engine/motion/embodiments/` | G2 双臂和二值夹爪；UR7e/Robotiq 关节夹爪；Pine 腕部相机碰撞描述；旧 Agibot 单独保留 |
| G2 任务采集 | `data_engine/g2/` | 任务配置、recipes、实测状态规划、分类工作台、物理成功审计、回放与 LeRobot 导出 |
| Pine 任务采集 | `data_engine/pine_wm/` | 首批 20 项配置、中央工作区布局、接触/持物检查、单次实时预览和审核状态 |
| 通用数据合同 | `data_engine/recording/`、`annotations/` | 控制时间对齐、视频检查、skill/stage/event 及 HDF5 标注 |
| 资产管理 | `data_engine/assets/` | HF 版本 pin、下载、完整性和依赖检查、机器人/场景/物体分开打包 |
| 运行查看 | `data_engine/viewer/`、`pine_wm/viewer/` | 通用运行报告页，以及原 Pine 20 项任务审核页 |
| 测试/文档 | `data_engine/**/tests/`、`test_*.py`、`docs/harness/` | 管线合同、兼容导入、包边界、物理专项与实际运行证据 |

公共模块不拥有具体任务的动作次序。G2 的 `collection/recipes.py` 与 Pine 的
`collection/primitives.py` 保留策略差异，共用底层规划与执行。
`CumotionEmbodimentCfg` 声明关节名、工具坐标、夹爪动作语义、碰撞模型；registry 延迟解析硬件。

## 留在 Arena 扩展接口上的内容

| 内容 | 位置 | 原因 |
|---|---|---|
| G2 机器人/相机/动作/recorder | `isaaclab_arena/embodiments/g2/` | Arena 实际创建机器人和动作管理器 |
| UR7e、Robotiq、Pine 标定/腕部硬件 | `isaaclab_arena/embodiments/ur7e/` | 同一 Arena 机器人也可被 teleop、策略评估或其他任务使用 |
| 场景工厂 | `isaaclab_arena_environments/g2*.py`、`*pine_wm*.py`、`ur7e*.py` | 场景构建沿用 Arena environment registry |
| 任务/成功谓词 | `isaaclab_arena/tasks/` | 物理成功由环境判定，不让规划器声称成功代替任务成功 |
| 外部资产接入 | `isaaclab_arena/assets/usdcraft_scene.py`、`g2_asset_paths.py`、`local_objects.py`、`geniesim*` | 将 HF ID 与外部模型解析成 Arena 资产 |
| bin 碰撞修复 | `assets/convex_decomposition.py`、`g2_clean_workcell_environment.py` | 在实际 spawn 时覆盖凸分解参数；源 USD 保持版本不可变 |

不另写一套 Scene、Robot、Task registry。业务代码归 Data Engine，Arena 注册接口继续位于
Arena 原生扩展位置。通用 `physics_config.py` / `physics_spawner.py` 等已存在于本次上游基线，
不把这些上游能力误列为我们的新增实现。

## 对已有 Arena 文件的主要补丁

- `environments/arena_env_builder.py`：仅在需要 placement IK 时加载 cuRobo，避免 native cuMotion
  采集隐式加载第二个后端；检查 env_cfg_callback 的返回值。
- `terms/events.py`：避免给固定/运动学对象写不适用的速度；补关节状态和驱动目标一起 reset 的入口。
- `utils/cameras.py`：补相机视点跟随机器人刚体的辅助函数。
- `assets/registries.py`、`device_library.py`、`retargeter_library.py`：注册外部资产和双臂输入设备。
- `embodiments/__init__.py`、`tasks/task_library.py` 等：接入新增机器人和任务；这些旧注册入口
  保持既有方式，新建 Data Engine 包的 `__init__.py` 不做 eager re-export。
- `tasks/predicates/spatial.py` 等：已有工作单元任务需要的姿态、容器、堆叠和稳定性谓词。

## 历史内容与兼容位置

- `isaaclab_arena_cumotion/*.py` 及原 `g2_collection/`、`tools/pine_wm/first20/`、相关 `tools/`
  现在是转发入口。业务实现只在 `data_engine` 一份；正常 import 返回同一模块对象。
- `isaaclab_arena_cumotion/scripts/` 保留旧 RR 和 Agibot 的专用驱动，已改用新的公共运动模块。
  它们不等于已经接入通用 Harness，更不能复制成 Pine 的任务。
- `isaaclab_arena/policy/ur7e_diffusion_policy_remote.py`、`isaaclab_arena_gr00t/lerobot/`、
  `docs/rr_sim2real/` 等是 RR 训练/评估与迁移历史，不是本轮新建管线。
- 当前分支相对官方 main 还包含 `pyproject.toml`、`uv.lock`、Docker 和 CI 的历史差异。
  本轮只在 pyproject 增加 `data_engine*` 包发现，没有修改 Docker、CI、共享环境或 submodules。

完整路径清单见 [upstream_paths_20260927.csv](upstream_paths_20260927.csv)。
它记录工作树相对该基线的差异，包含 untracked 新源码，不代表本轮全部新增，也不是待提交清单。

## 验证范围与后续边界

本轮调整目录和依赖方向，不改变任务控制周期、夹爪目标、成功阈值或采集时序。
新旧 CLI 等价、同模块身份、懒加载和来源覆盖均有回归检查。
实际验证结果见 [开发记录](development.md)。

同步 pre-action 取图、全任务细粒度标注、严格 planner CUDA 证明、稳定性/批量调度仍是后续步骤；
不能因目录集中就宣称它们已经统一。旧 run 的原路径和哈希不改写，复查应使用对应源码版本。
