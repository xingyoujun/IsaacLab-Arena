# 现有实现与整理归属

基线见 [索引](README.md)。这是整合前的归属审计：`local` 表示原工作树存在；`remote` 表示来自
`09a786fb4` 的新增内容。开发启动后该提交已经整合，新运行证据见 [开发记录](development.md)。
下表是迁移归属，不表示所有目标重构均已完成。

| 内容 | 当前来源/位置 | 状态与复用决定 |
|---|---|---|
| 环境组合 | local `isaaclab_arena/environments/arena_env_builder.py`、`arena_environment_factory.py`、`arena_world.py` | 复用工厂/组合机制；增加采集运行适配，不重写模拟器 |
| Embodiment 基类 | local `isaaclab_arena/embodiments/embodiment_base.py` | 保留；profile 引用现有 embodiment，补控制/传感器能力声明 |
| Scene / Task / registry | local `scene/scene.py`、`tasks/task_base.py`、`assets/registries.py` | 保留；新增轻量 scenario 组合描述，不复制一套注册逻辑 |
| 动作/状态 recorder | local `isaaclab_arena/terms/recorders.py`、UR7e `demo_recorders.py` | 作为 recorder hook 接入基础；机器人通道映射留适配层 |
| Episode JSONL | local `recording/episode_recorder_manager.py` | 可复用统计 term；当前初始化输出会清空目标，不能直接当作可恢复批次账本 |
| Policy / scheduling | local `policy/policy_base.py`、`policy/action_scheduling/` | 复用动作来源生命周期；补实际执行 chunk、时延、取消和控制步关联 |
| Experiment runner | local `evaluation/experiment_runner.py`、`run_execution.py` | 复用运行配置/清理经验，与评估保持边界；不复制第二套底层 env 主循环 |
| Variation / relations | local `variations/`、`relations/` | 复用 sampler/布局约束；补实际样本、拒绝分布及资格版本 |
| PhysicsBackend | local `utils/physics_backend.py` | 存在 PhysX/Newton；harness 每种物理/对象组合仍需验证 |
| 公共 cuMotion | local + remote `isaaclab_arena_cumotion/planner.py`、`executor.py` | 保留规划后端；支持二值夹爪等远端改进；阻止绕过记录通道 |
| G2 控制与 recorder | remote `embodiments/g2/`、`isaaclab_arena_cumotion/g2.py` | 合入后作为 G2 adapter，不把关节名或 16D 写入核心 |
| G2 Session / recipes | remote `g2_collection/session.py`、`recipes.py`、`workcell/` | 拆分机器人控制、任务 recipe、调度职责；现有实现先经兼容适配保留 |
| G2 CLI / audits | remote `tools/data_collection/g2.py`、`g2_collection/validation.py` | 吸收分阶段执行/独立审计；移除通用层对三任务与三相机的硬编码 |
| G2 export | remote `tools/data_collection/export_g2_episode.py`、`g2_collection/metadata.py` | 复用原子提交和来源绑定；字段模式由 profile 提供 |
| Pine WM 20 项 | local `tools/pine_wm/first20/`、`pine_wm_first20_environment.py` | 保留已审布局/判据；拆出 adapter 后逐项迁移，禁止复制到 G2 目录 |
| Pine 工作区与碰撞 | local `first20/geometry.py`、`contacts.py`、`review_layout.py` | 核心可复用算法与具体工作区/工具参数分离，保留腕部相机保护 |
| Pine dashboard | local `tools/pine_wm/dashboard/` | 保留用户入口，后续改读统一 run index；不重新推断任务完成状态 |
| RR 数据链路 | local `isaaclab_arena_cumotion/scripts/collect_ur7e_*` 与 `ur7e_*` | legacy adapter；保持旧任务/资产方法/数据契约，不混入 Pine |
| 时间合同 | remote `isaaclab_arena/recording/alignment.py` | 复用 pre-step 映射/视频检查；拆出写死 robot 名、UR/G2 joint 分类、单位和阈值 |
| 回放渲染 | remote G2 `workcell/rerender_states.py` 与公共 `rerender_embodiment_cameras.py` | 合并协议不盲合实现；分别声明 bounded/exact、相机时钟、来源哈希 |
| Skill trace | local 未提交 `isaaclab_arena/annotations/` | T001/T041 实测成功；推广 actor/对象映射与 exporter 保留，未知标签不伪造 |
| 资产管理 | local + remote `tools/usdcraft_scene/`、`assets/usdcraft_scene.py` | 保留 Git/HF 分工；新增 G2 版本未在本机下载验证 |
| LeRobot / DP | local + remote `isaaclab_arena_gr00t/lerobot/` | 统一 normalized episode 输入；旧格式只读兼容，新格式严格对齐 |
| 历史 Agibot / 其他任务 | local 既有环境/任务库 | 登记待适配，不能因共用 Arena 就标为 harness 已支持 |

## 任务家族与证据

| 家族 | 已知范围 | 当前证据 | 不能据此宣称 |
|---|---|---|---|
| Pine WM | 200 项规划目录；优先 20 项实现 | 本机 20 项单成功布局预览；T001/T041 分别 419/280 步标注试验 | 200 项可采集；当前版本 20 项稳定性通过 |
| G2 | stack_bowls、peg_into_sleeve、clean_workcell_table | 远端文档记载各一条原生 cuMotion 完整 pilot；本机未复跑 | 本机通过；随机化批量可靠；严格 CUDA 通过 |
| RR real2sim | 抽屉/按压/旋钮与四种资产方法，具体映射见 RR 文档 | 历史数据/训练/评估有独立版本与记录 | 已按新时间/标注合同全量迁移 |
| Arena 其他任务 | 现有 registry 中的环境、机器人与物理后端 | 只有原项目各自的测试/运行证据 | 自动获得本 harness 的采集、回放、标注和导出资格 |

## 已知整合点与风险

1. 本地与远端同时修改 `tools/pine_wm/first20/dataset.py`：需要保留远端 transition contract
   和本地 skill trace 两种元数据，验证同一转移索引。不能选择一方覆盖。
2. Pine 在线四路预览为 post-step；远端训练导出配置选三路 pre-step。第四路第三人称是审核流，
   不应为了数量一致伪造/删除相机；由 capture/export profile 决定。
3. `alignment.py` 的 ReplayDrift 按机器人关节名称判断相机链，且不同 joint 坐标混用统一阈值。
   通用化必须按 robot/state codec 声明关节类型、单位与相机链，特别是移动关节不能按弧度检查。
4. G2 exporter 有 raw/video SHA 关联；通用 LeRobot 路径仍需补等强度来源检查和标注透传。
   仅帧数与 fps 相等不能证明视频来自该 episode。
5. G2 `phase(name)` 是日志边界，不等于通用子阶段规范；叠碗中的闭合动作等需在执行语义边界补标注。
6. G2 bin 的历史 CPU collision fallback 已完成冷烘焙与接触回归修复（见 bin_collision_20260927.md）；planner device 仍未独立证实，严格 GPU profile 保持未通过。
7. G2 native 任务是固定布局，seed 不能替代随机化；历史 cuRobo 批次成功率不能用于新原生路径。
8. 远端 launcher 面向 Docker，本机 native 且共享 venv；不能在本工作树安装/升级共享运行环境。
9. HF 本地原 pin 为 `d0bfcf1fcda19753569e222c834386509e40e1b5`，整合后的新 pin 为
   `082b4f1164ddd47ef5f0e775bdfeb14401691575`。开发阶段已独立下载校验 89 个 manifest 文件约 132 MB；
   旧缓存保留，旧 run 继续引用原资产版本。

## 内容整理规则

- 现在：以本目录建立统一入口、归属和迁移清单；保留所有正在使用的路径与旧数据身份。
- 适配时：新公共逻辑进入 collection/recording/annotations，任务特有逻辑仍归各任务家族。
- 迁移后：旧 CLI 先转为明确的兼容入口；确认无消费者后删除，历史内容查 Git，不复制 backup 目录。
- 原始数据/模型/视频在外部或 gitignored 路径；HF 只承担资产发布，不混入训练集。
- 不移动 sibling checkout 的内容，不借整理修改共享 venv、submodules、Docker 或 CI。

远端源码可按固定提交检查，例如：
`git show 09a786fb4:isaaclab_arena/recording/alignment.py`。
远端完整说明：`docs/g2_development.md`、`docs/g2_native_release_20260926.md`，整合后可在本工作树读取。
