# USDCraft-Scene 资产管理

2026-09-30 v2 交互标注接入见 [验证记录](../../docs/harness/usdcraft_v2_20260930.md)。
`python -m data_engine.assets.interactions --assets BUNDLE --output REPORT.json` 可批量核对
USD 内嵌交互与 hash 绑定的 sidecar；harness preflight 对所用的有标注资产自动执行相同检查。

私有 HF 仓库：[xingyoujun/USDCraft-Scene](https://huggingface.co/datasets/xingyoujun/USDCraft-Scene)。
代码、标定、管理脚本和 `release.json` 放 Git；USD、网格、纹理放 HF。

## 统一的本地目录

所有人统一下载到 **本仓库根目录下的 `local_assets/USDCraft-Scene/`**。
根 `.gitignore` 已忽略 `/local_assets/`，里面的模型及 HF 下载元数据不会进入 Git。
工具通过自身文件位置定位仓库，因此从其他工作目录执行也不会下载错位置。

```text
IsaacLab-Arena-tasks/                 # Git checkout，目录名可以不同
  local_assets/                     # Git 忽略，不做备份副本
    USDCraft-Scene/
      manifest.json
      embodiments/pine_wm/robot/     # UR7e、Robotiq 与完整 USD 依赖
      embodiments/pine_wm/calibration/
      scenes/pine_wm/scene.json      # 程序化桌面、灯光的场景描述
      scenes/pine_wm/review_layouts.json
      assets/cube_40mm/model.usdc    # 公共物体，不归属于某个场景
      assets/drawer_first20/model.usdc
      assets/…
  tools/usdcraft_scene/
    manage.py
    release.json                    # HF commit + manifest SHA-256，随代码同步
```

新增机器人放 `embodiments/<名称>/`，新增场景放 `scenes/<名称>/`，公共物体放
`assets/<稳定名称>/`。不要使用 `backup`、`final_v2` 或日期目录维护资产版本；版本由 HF history 管理。
原始压缩包、截图、仿真视频、HDF5、训练数据和凭据不放进资产包。

## 下载与使用

先登录有私有仓库读取权限的 HF 账号（`hf auth login`；不要把 token 写入代码或命令示例）。
在本仓库对应的 Arena 容器中，从仓库根目录执行：

```bash
/isaac-sim/python.sh tools/usdcraft_scene/manage.py download
/isaac-sim/python.sh tools/usdcraft_scene/manage.py verify
/isaac-sim/python.sh tools/usdcraft_scene/manage.py audit-usd
```

以上命令默认操作 `local_assets/USDCraft-Scene/`，Pine WM 仿真也默认读取这里。
下载锁定 `release.json` 中的不可变 HF commit，不自动跟随 `main`；下载后逐文件校验 SHA-256。
首次下载需要联网，仿真运行不会隐式下载。

如以前配置过路径，先清除覆盖项以恢复统一目录：

```bash
unset ARENA_USDCRAFT_SCENE_ROOT ARENA_PINE_WM_ASSET_ROOT ARENA_PINE_WM_FIRST20_ROOT
```

只有确实需要外置磁盘时才覆盖根目录，并把相同路径传给管理命令：

```bash
export ARENA_USDCRAFT_SCENE_ROOT=/your/external/disk/USDCraft-Scene
/isaac-sim/python.sh tools/usdcraft_scene/manage.py download
```

显式的旧 `ARENA_PINE_WM_ASSET_ROOT` / `ARENA_PINE_WM_FIRST20_ROOT` 仍可读取历史资产包。
资产的重命名应更新 manifest 的稳定 ID 映射和所有 USD 相对引用，不能只移动最外层 USD。

## 修改与上传

日常校验和上传也使用同一个本地目录：

```bash
/isaac-sim/python.sh tools/usdcraft_scene/manage.py verify
/isaac-sim/python.sh tools/usdcraft_scene/manage.py audit-usd
PYTHONPATH=. /isaac-sim/python.sh tools/usdcraft_scene/stage_release.py \
  --source local_assets/USDCraft-Scene --output local_assets/releases/g2_next
/isaac-sim/python.sh tools/usdcraft_scene/manage.py upload \
  local_assets/releases/g2_next/payload --release-plan local_assets/releases/g2_next/release-plan.json
```

上传会验证现有 manifest，仅上传清单内文件，要求目标仓库保持私有，并将返回的
HF commit 和 manifest 校验值写入 Git 侧 `release.json`。不会删除远端的其他资产。
**不要在未更新 manifest 的情况下修改模型后上传**：校验会拒绝内容不一致的文件。
新增资产需填写 entry、相对路径、来源、文件大小和 SHA-256；新增其他场景时需合并完整仓库清单。
上传器发现远端存在清单之外的文件会停止，防止覆盖后续其他人添加的资产。
G2 机器人、房间、物体和规划描述与 Pine WM 共用完整资产清单。

从原始来源首次组包使用 `prepare`，仅在目标目录不存在时执行；它不会覆盖现有下载，也不会创建备份：

```bash
/isaac-sim/python.sh tools/usdcraft_scene/manage.py prepare \
  --scene-source /path/to/extracted/scene \
  --objects-source /path/to/extracted/first20
```

`prepare` 默认生成同一个 `local_assets/USDCraft-Scene/`。一般协作者只需 `download`，不需要原始包。

## RR real2sim 与 Pine WM 的边界

两套任务独立管理，见 [任务归属](../../docs/task_families.md)。
`manage.py prepare` 的首次组包范围是 Pine WM 机器人和首批 19 种物体，不读取 RR 任务或资产目录。
已发布的共享包另外包含 G2；日常开发使用锁定版本下载，无需再次从旧目录组包。
RR 的 USDA 适配层保留在 `tools/rr_sim2real/asset_overlays/`，其运行入口也不变。
Pine WM 的 T041–T045 使用 `P20_drawer`，不是 RR 的 `drawer_rr`。

已发布的首个 HF revision 中仍有 `assets/usdcraft_drawer_rr/`，它是早期交叉验证留下的
独立 RR 公共资产；当前 Pine WM 任务不引用它。为保留已发布版本的一致性，本次只整理
代码归属，不删除远端文件或改写旧 release。后续发布需显式合并完整清单，不能把 RR 任务
纳入 Pine WM，或把 RR 专用 USD 一并移动进 `scenes/pine_wm/`。

## 代码与运行依赖

- `pine_wm_ur7e`、`pine_wm` 和 `P20_*` 是稳定 ID；长生成记录名仅保留作来源记录。
- 桌子、灯光、打印支架和夹爪材质由 Arena 代码生成，HF 不包含已烘焙的独立完整场景。
- Git 侧相机标定是运行时来源；HF 同时包含对应标定，发布时需要保持一致。
- 机器人材质依赖 Isaac Sim 的 `OmniPBR.mdl`；离线检查允许这一运行时依赖，几何必须全部位于包内。
- 首批任务语义以 Git 的 `data_engine/pine_wm/tasks.json` 为准，HF manifest 负责解析物体 ID。
- 资产迁移已做文件校验和 USD 依赖检查，不代表已在全新机器上重新完成全部仿真测试。

Pine WM 工具输出默认放 `outputs/pine_wm/first20/`（同样被 Git 忽略），可用
`ARENA_PINE_WM_EXPERIMENT_ROOT` 指向已有实验目录，网页及监控读取相同变量。
非标准 Isaac Lab 安装可设置 `ARENA_ISAACLAB_SOURCE`；串行预览保留已有 `PYTHONPATH`。
资产路径统一不改变任务的审核状态：当前只完成单次成功预览，稳定性测试和采集仍未开放。

## Git 同步检查

同步代码、标定、规划描述、任务配置、本 README、管理脚本和 `release.json`。
不要同步 `local_assets/`、`outputs/`、缓存、视频或数据集。提交前可只读检查：

```bash
git status --short
git diff --check
git check-ignore local_assets/USDCraft-Scene/manifest.json
```

管理命令不会执行 Git 暂存、提交或推送。

## G2 发布

G2 使用[统一开发与采集流程](../../docs/g2_development.md)。
`stage_release.py` 从完整已发布资产包生成候选版本，刷新 Git 中的任务、规划参数和标定快照。
候选目录的 `release-plan.json` 锁定清单哈希与远端父提交；上传前必须通过 `verify` 和 `audit-usd`。
`manage.py upload PAYLOAD --release-plan PLAN` 拒绝过期父提交，成功后更新 Git 的 `release.json`。

`prepare_g2.py` 仅用于从原始供应商文件首次引入 G2；房间和铝块通过显式 `--room`、
`--aluminum-stock` 输入。日常开发者直接下载已发布版本，不需要旧数据目录。
资产发布不包含采集 HDF5、视频、训练集或模型权重。
