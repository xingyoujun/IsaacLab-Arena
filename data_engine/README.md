# Data Engine Harness

Data Engine 是建立在 Isaac Lab-Arena 上的仿真数据生成管线。Arena 提供环境组合、机器人、
物理步进、动作管理和 recorder 接口；本模块负责采集规划、执行编排、数据合同、标注、
审核与产物管理。既不复制 Arena 的环境注册系统，也不把某一种机器人流程当成公共标准。

## 当前源码结构

```text
data_engine/
  cli/harness.py              # 统一 list / preflight / plan / preview / status / audit
  harness/                    # 生命周期、GPU lease、版本身份、质量检查
    adapters/                 # 家族入口、能力与验收适配
  motion/
    cumotion/                 # G2、Pine 共用的 planner / executor / grasps / robot_description
    embodiments/
      g2.py                   # G2 双臂、锁定关节、TCP、二值夹爪适配
      ur7e_robotiq.py          # UR7e + Robotiq；Pine 腕部硬件碰撞配置
      agibot_legacy.py         # 保留旧 Agibot 描述，和 G2 分开
  recording/alignment.py      # observation/image/action/next_state 对齐与视频检查
  annotations/                # 通用 skill / stage / event 及 HDF5 标注
  assets/                     # HF 下载、完整性检查、打包、发布与版本 pin
  g2/
    tasks.json                # 任务及环境参数
    clean_workcell_table.yaml # 场景布局、资产绑定、成功条件
    workcell_motion.yaml      # 抓取候选与动作参数
    cli.py                    # G2 家族采集／回放／导出入口
    collection/               # recipes、session、workcell 与物理成功检查
    export.py                 # G2 数据合同对应的 LeRobot 导出
    check_bin.py              # bin GPU 碰撞专项检查
  pine_wm/
    tasks.json
    review_layouts.json
    collection/               # qualify、primitives、布局、接触检查与实时预览
    viewer/                   # 原 Pine 任务网页、审核队列与监控
  viewer/                     # 通用任务审核台、尝试命名与视频阶段联动
  tests/                      # 模块边界、兼容导入和来源追踪测试
```

两个家族引用相同的 `CumotionArmPlanner`、`EnvActionExecutor`。关节名、TCP、夹爪的
joint-position / binary 语义、碰撞描述由 embodiment profile 决定。任务动作序列仍在各家族
`collection/`，不能因为名称相似就合并不同的抓取或接触策略。
Profile 延迟加载；列任务不会启动 Isaac Sim，加载 G2 配置也不要求先加载 UR7e 资产。

Pine WM 是历史工作单元名称，实际 embodiment 是 UR7e + Robotiq + 腕部相机。
目录按当前家族归档，运行模型仍分别引用 embodiment、scene、task、asset，后续不必为
每个机器人/场景组合复制管线。

## Arena 集成位置

- `isaaclab_arena/embodiments/g2/`、`embodiments/ur7e/`：Arena 原生机器人、动作和相机配置。
- `isaaclab_arena_environments/g2*.py`、`*pine_wm*.py`：Arena 环境工厂及任务场景构建。
- `isaaclab_arena/assets/`：运行时资产解析、spawn physics 和必要的 Arena 集成。
- `isaaclab_arena/tasks/`：Arena 原生任务及终止谓词。

这些位置是仿真接口，继续使用 Arena 的扩展机制。采集调度和规划后端实现归本模块。
模型／纹理由私有 HF `xingyoujun/USDCraft-Scene` 管理，下载到 gitignored `local_assets/`；
raw 数据、视频和实验报告留在 gitignored `outputs/`，不放进 Python 包或 Git。

## 入口与兼容

设置本工作树的 native PYTHONPATH 后，使用：

```bash
.venv/bin/python -m data_engine.cli.harness list
.venv/bin/python -m data_engine.cli.harness explain --scenario g2/stack_bowls
.venv/bin/python -m data_engine.g2.cli list
.venv/bin/python -m data_engine.assets.manage --help
python3 -m data_engine.viewer.server --port 8091
```

运行目录可附加 `diagnostic_review.json`，用相同 `spec_id` 绑定补充诊断。
查看页会突出显示 `warning` / `blocked` 及 `summary`，但不改写已封存 payload、
原始验收结果或训练资格。这适用于任务预览通过后发现的动作／碰撞问题；
诊断证据与源码位置见 [G2 单次复核](../docs/harness/g2_recheck_20260927.md)。

旧的 `isaaclab_arena_cumotion` 公共模块、`isaaclab_arena.collection`、`annotations` 及相关
`tools/` 入口只转发到这里；普通导入获得同一个模块对象，不保留第二份实现。
`__init__.py` 不做 eager re-export。`pyproject.toml` 已加入 `data_engine*` 包发现。
RR 和旧 Agibot 的专用历史脚本仍留在 `isaaclab_arena_cumotion/scripts/`，它们也引用新的
公共规划模块；本次不把它们伪装成已验收的通用 Harness adapter。

## 尚未统一的行为

代码集中不代表行为已全部统一：Pine 当前是 post-step 实时预览；G2 当前是 states-first
加有界 pre-step 回放。后续默认目标是同步记录 `observation[t], image[t], action[t], next_state[t]`，
离线渲染为可选派生产物。批量资格、细粒度标注全覆盖和 native planner CUDA 证明仍按质量门槛推进。

[任务准备与抓取选择](../docs/harness/task_preparation.md) ·
[使用与当前能力](../docs/harness/usage.md) ·
[相对上游 main 的改动](../docs/harness/upstream_diff_20260927.md) ·
[HF 资产管理](assets/README.md)

USDCraft v2 的 body-local grasp/interaction、热水壶/烤面包机联动场景与实测限制，见
[2026-09-30 接入记录](../docs/harness/usdcraft_v2_20260930.md)。入口为
`interactions/kettle_release`、`interactions/toaster_cancel`；需显式选择包含新资产的 bundle。
标注候选通过静态审计不等于机器人执行、随机稳定性或训练数据资格。
