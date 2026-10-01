# Harness 开发与验收记录

本轮开始实施：2026-09-26。公共管线建立在 Arena 上，不新增环境、机器人或任务框架。
只用现有 Arena 工厂、采集器及 recorder；任务内执行算法不由管线重新实现。

## 开发路径

- `data_engine/harness/`：纯 Python 配置解析、运行身份、阶段管理、来源/质量检查。
- `data_engine/harness/adapters/`：既有 Pine/G2/RR 路径的薄适配；按能力报告支持程度。
- `tools/data_collection/harness.py`：统一 CLI，与现有 G2 入口兼容共存。
- `data_engine/harness/tests/`：合同、错误路径、恢复和适配器测试。
- `docs/harness/`：设计、命令、开发记录、验收证据索引。
- `outputs/harness/`：本机试验/运行报告，不进 Git；模型仍在 `local_assets/`。

首版是已有采集流程的管线整合，不声称已抽离所有任务 recipe、实现所有机器人、分布式或批量稳定性。
Native runtime 使用本工作树 PYTHONPATH，不更改共享 venv。

## 逐步验收

| 步骤 | 开发内容 | 必须验证 | 状态 |
|---|---|---|---|
| D0 | 整合远端，保留本地标注 | dataset 双合同、资产新旧隔离、无数据入 Git | 完成 |
| D1 | scenario 引用/解析、不可变配置身份 | 无 Sim 导入、非法引用、配置变化失效、动作维度不混淆 | 首版通过 |
| D2 | stage 生命周期、日志、锁、重试/取消 | 不覆盖产物、并发锁、失败状态、恢复前来源检查 | 基础测试通过；仅恢复审计 |
| D3 | Pine/G2 薄适配、统一质量审计 | 真实 raw/video/hash/annotation，preview 不冒充训练图像 | Pine 两项、G2 插销通过 |
| D4 | CLI、运行索引与可查看报告 | list/plan/preflight/preview/audit 命令，未知 CUDA 不授予资格 | 首版通过 |
| D5 | 新 episode 实测 | 单次成功即停、失败保留、逐帧检查、场景不变 | 3 个成功 pilot、1 个失败，未做稳定性 |
| D6 | 文档与兼容回归 | 格式/单测、旧网页保留、明确未支持项与后续门槛 | 首版验收完成，范围见下文 |

数据管线的集成验收由本轮自主完成；布局审核与批量稳定性资格仍需用户确认。
GPU 严格资格不能通过修改阈值或把 unknown 改为 pass 取得。

## 本轮证据

- `outputs/harness/pine_t001_001`：新入口实际成功；419 个控制步，四路 15 fps 视频，
  2 个 skill / 12 个 stage，完整视频解码与 HDF5/标注校验通过。worker 用时 118.26 s。
  这是 post-step 预览；training_eligible=false，未授予稳定性资格。
- `outputs/harness/g2_stack_001`：保留一次失败。右臂下降末端误差 2.373 mm > 原任务阈值 2 mm，
  未改阈值、未自动重试、未导出成成功样本。证明远端 pilot 不替代本机任务验证。
- `outputs/harness/g2_peg_001`：555 步；raw 审计、三路全视频解码和 LeRobot 导出通过。
  worker 完整采集/渲染/导出用时 592.67 s；不等于仅采集耗时。尚无 skill trace 适配，标为 unknown。
- `outputs/harness/pine_t041_001`：第二轮控制步观察器接入后新执行，280 步、四路视频、
  1 个 skill / 11 个 stage（含一个零长度拒绝阶段），worker 用时 83.56 s。
  四路相机缓冲实际报告 cuda:0，不能据此推断 native planner 也已通过严格 CUDA 验证。
- 新 HF 包验证 89 文件 / 132,291,408 bytes；对比旧 manifest，原 entries 均保持，
  旧文件仅 README / LFS 元数据改变，Pine 资产 payload 未变。
- 最终管线/标注单测 18 项、G2 CLI 6 项、时间合同 9 项、G2 其他相关纯代码测试 19 项通过，
  共 52 项。未运行整个 Arena 的完整仿真测试套件。
- 查看页 `tools/harness_viewer/server.py`，本机端口 8091；根据 manifest 展示状态，
  Pine 四路视频 HTTP byte ranges 和页面 JavaScript 语法已检查。原 8090 网页保留。
- `outputs/harness/acceptance_20260926.json`：最终 validator 对三份封存 pilot 的只读回归，
  记录 producer spec ID 和本次 validator 源哈希，不覆盖原 run/quality，也不伪装成新物理资格。

## 实现边界与迭代结果

- 公共 runner/storage/steps/audit 不包含机器人关节或相机数量；家族目录、命令和证据映射在 adapters。
- Pine 使用一个 `ControlStepObserver`，原 Arena action manager/recorder 继续记录；preview 不再自己包裹 env.step。
- 完整物理步后编码失败仍保留该动作标签，并写 capture_failed 事件；不完整物理步不伪造样本。
- 执行前后检查配置/源身份，结束后重检资产字节；封存后重审拒绝任何 payload 变化。
- 超时/取消只清理自己创建的进程组；启动器失败后遗留子进程的回收有专项测试。
- G2 插销关键画面已人工查看：腕部可见操作，头部画面部分被手臂遮挡。没有修改标定或将其
  标为“所有视角无遮挡”。Pine 两项末帧保持原工作区与第三视角布局。

## 首版之后仍需开发

1. G2 精细 skill trace 与独立 actor/channel 语义；当前不会把粗阶段日志冒充细标注。
2. Pine pre-action 训练图像/统一 exporter 与合法失败轨迹的数据集选择规则。
3. 严格 CUDA 证据、G2 叠碗本机跟踪误差、工作台 bin 的 CPU collision fallback。
4. 版本化人工审核与稳定性资格的完整操作接口、随机化分布/覆盖验证。
5. RR 薄适配、其余 Pine 场景资格、阶段级渲染/导出恢复、批量及多 worker。

本轮完成的是可运行、可审计的数据管线首版，不是整个 roadmap 或所有场景的生产资格。
真实运行分别绑定当时源哈希；开发期间后续纯管线改动由只读回归单独记录。


## 2026-09-27：配置集中与 bin 碰撞修复

任务配置与布局统一进入 `data_engine/pine_wm/`、`data_engine/g2/`；原文件迁移，未保留副本。
CLI、环境、网页、HF 打包工具和 source hash 同步更新。HF 已发布快照不修改。
G2 bin 专项复现、修改与验证记录见 [bin_collision_20260927.md](bin_collision_20260927.md)。
原 D6 的 bin CPU fallback 项已完成专项修复；native planner CUDA 证明与完整任务稳定性仍待完成。
同步 pre-action 图像采集是后续统一方向，本次没有宣称 G2 已改用 live capture。


## 2026-09-27：Data Engine 模块归并

公共规划实现迁入 `data_engine/motion/cumotion/`，硬件配置拆到
`motion/embodiments/{g2,ur7e_robotiq,agibot_legacy}.py` 并延迟加载。
管线、标注、对齐、资产管理、两个家族采集器和网页归 `data_engine/`；旧入口仅转发。
Arena 环境工厂、机器人和 recorder 集成仍沿用原生扩展位置。

- 75 项相关测试通过，包括同模块身份、CLI 等价、无仿真导入、来源覆盖及家族回归。
- Pine T041：`outputs/harness/pine_t041_module_20260927`，单次成功，worker 81.82 秒，
  280 控制步，四路 CUDA 相机预览与阶段标注通过 Harness 检查。
- G2：完整工作台初始化，左右臂使用同一公共 planner，分别规划 0.01 rad 关节位移；
  FK 位置误差左 0.000443 mm、右 0.001219 mm。不是整条 G2 任务的重新采集。
- 8090 和 8091 首页/API 均 HTTP 200；旧 Pine 网页仍显示 20 个任务。
- pre-commit 与 git diff --check 通过；没有运行完整 Arena 三阶段套件。
- 证据索引：`outputs/harness/module_refactor_20260927/acceptance.json`。

本轮仅整理源码和接口边界。两套相机采集时序、已知 planner CUDA 资格和单次审核门槛不变。
没有 commit/push，也没有改共享环境、Docker、CI 或主克隆的工作树。
