# 开发版 CLI

Harness 管理 Arena 采集器的运行与数据验收，不替换 Arena 环境/机器人注册。
当前阶段是单次 preview，沒有 `collect 200` 或自动授予稳定性资格的入口。

## 本机运行

本工作树使用已有 native venv，不安装依赖或修改共享环境：

```bash
cd /home/ubuntu/code/IsaacLab-Arena-tasks
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y
export PYTHONPATH="$PWD:/home/ubuntu/code/IsaacLab-Arena/submodules/IsaacLab/source/isaaclab"
.venv/bin/python -m data_engine.cli.harness list
```

新 G2/Pine 共享资产版本已独立下载到
`local_assets/releases/082b4f1164ddd47ef5f0e775bdfeb14401691575`，不会覆盖原默认缓存。

```bash
.venv/bin/python -m data_engine.cli.harness preflight \
  --scenario pine_wm/T001 \
  --assets local_assets/releases/082b4f1164ddd47ef5f0e775bdfeb14401691575

.venv/bin/python -m data_engine.cli.harness plan \
  --scenario g2/stack_bowls \
  --assets local_assets/releases/082b4f1164ddd47ef5f0e775bdfeb14401691575 \
  --output outputs/harness/stack_plan.json
```

`list` 不导入 simulator / torch；`preflight` 验证配置引用和资产字节，不能证明可达性、无碰撞或 CUDA。
`plan` 保存解析结果用于审阅；`preview` 重新解析当前配置，并把实际使用的 spec 固定在 run.json。
不隐式下载资产，不根据名字寻找其他机器人文件作为 fallback。

```bash
.venv/bin/python -m data_engine.cli.harness preview \
  --scenario pine_wm/T001 \
  --assets local_assets/releases/082b4f1164ddd47ef5f0e775bdfeb14401691575 \
  --run-dir outputs/harness/my_unique_preview --timeout 1800

.venv/bin/python -m data_engine.cli.harness status --run-dir outputs/harness/my_unique_preview
.venv/bin/python -m data_engine.cli.harness audit --run-dir outputs/harness/my_unique_preview
```

一次命令只执行一次尝试。目标目录已存在即拒绝；失败保留，不自动重试。Ctrl-C 或超时会停止该
worker 进程组；不会停止用户其他进程。每个 CUDA device 有本用户跨工作树 harness lease，
不代表可以检测所有非 harness GPU 进程，仍应与其他训练/仿真错峰。

当前 Pine 20 项均可通过 harness 启动单次 preview；T001/T041 有完整 skill trace，
其他 18 项保留未接入阶段标注的状态。原先批量/稳定性审批开关不被修改。G2 委托已整合的原生采集入口，支持情况以本机开发验收记录为准。
RR 保持旧入口和数据契约，尚未提供 harness 执行适配。

## 每次运行的内容

```text
run/
  run.json       解析配置、身份、阶段事件、封存产物哈希
  worker.log     完整子进程日志
  payload/       原采集器的 HDF5、图像/视频、任务报告及原生派生产物
  quality.json   公共数据合同、任务结果、图像用途、GPU 与审核状态
```

成功 worker 结束后封存 payload 的文件哈希，再进行独立检查。添加、修改或删除封存文件后重审会拒绝。
`audit` 只重做封存数据检查，不再启动仿真；来源代码改变时要求使用匹配版本，不静默套用新规则。
失败/被中断的 simulator 不支持从轨迹中间恢复，应使用新 run 目录发起新尝试。

`preview_validated` 表示 raw 和预览通过当前自动检查，仍待人工查看，不等于训练资格。
Pine post-step 审核视频不标为 pre-action 训练图像；G2 bounded replay 保留任务容差待验收状态。
没有独立 CUDA 证据时 GPU 状态是 unknown，`training_eligible`、`batch_enabled` 保持 false。
G2 原生入口生成的 LeRobot 放在 payload 中保留，其存在本身不授予 harness 正式数据发布资格。

## 开发检查

```bash
PYTHONPATH="$PWD" .venv/bin/python -m unittest data_engine.harness.tests.test_pipeline
PYTHONPATH="$PWD" .venv/bin/python -m unittest data_engine.annotations.test_skill_trace
```

测试只验证数据管线逻辑；实际物理、相机与任务结果必须使用完整 pilot 检查。
大文件与运行结果留在 gitignored `outputs/`，不进入 Git 或资产 HF 仓库。

## 运行查看页

```bash
python3 -m data_engine.viewer.server --host 127.0.0.1 --port 8091
```

默认读取本仓库 `outputs/harness/*/run.json`，只开放索引报告和封存预览视频。
本机开发服务在 8091 端口，旧 Pine 页面仍在 8090。网页不修改审核决定或启动任务。

8091 按任务归组，同一任务的尝试历史默认折叠；每 10 秒自动读取新结果，保留当前视频位置。
在任务中选择尝试，点击“修改名称”保存可读标题。名称写入 run 目录的
`viewer_metadata.json`，绑定 spec_id，不重命名目录或修改封存的 payload/原始验收结果。

播放器按真实记录的阶段区间高亮当前子任务和动作，点击阶段或拖动进度条可跳转，
切换相机保留时刻和播放状态。零步规划事件独立折叠，不占视频时长。
没有阶段记录的历史尝试只显示整体进度，不自动生成阶段标签。

同源 `POST /api/attempt-name` 仅允许修改显示名称；`/api/tasks` 为任务归组数据，
`/api/runs` 保留运行级数据和旧 run 名称锚点兼容。

## Pine WM 新资产全量同步（2026-09-30）

8091 首页「Pine WM · 全部任务同步」入口：`/reviews/pine-wm`；
JSON 为 `/api/pine-tasks`。20 个已有 Pine cuMotion 流程加两个 appliance 联动任务，
共 22 项；页面按 `USDCraft-Scene-v2-20260930/manifest.json` 的 SHA256
匹配实际 run 的资产版本，显示最新尝试（包括失败和待运行），不借用旧版成功状态。

本轮授权范围是把全部已有任务用新资产同步到看板，每项单次功能预览；不启动
稳定性或批量采集。运行目录 `outputs/harness/pine_*_v2_sync_20260930_01`，
队列状态 `outputs/pine_harness_sync_20260930/summary.json`。每次 worker 都独立保存源码
和资产身份、原始录制、结果和四路 post-step 视频；只有审计通过的预览可播放。
