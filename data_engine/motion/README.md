# 运动规划与机器人适配

`cumotion/` 是 Harness 的共享 native cuMotion 后端。当前 Pine 和 G2 使用同一份
`planner.py`、`executor.py`、`robot_description.py` 和抓取坐标约定。NVIDIA 的 cuMotion
实现仍来自 Isaac Sim 扩展，本目录不复制求解器本体。

`embodiments/` 保存硬件相关的配置工厂。公共 registry 提供
`register_cumotion_factory(name, factory, arm)`；factory 接收 `(env, arm)`，返回
`CumotionEmbodimentCfg`。硬件描述延迟加载，重复注册会失败。

新增机器人应提供关节组、TCP/工具坐标、动作及夹爪语义、规划碰撞模型，再注册适配。
新增任务应在对应家族的 `collection/` 写动作序列与成功检查，不在公共 planner 中添加 task ID 分支。
共享 `EnvActionExecutor` 通过 Arena action manager 执行，使 recorder 能看到实际动作。
旧 `ArmExecutor` 和 `PickAndPlace` 为历史调用保留，不自动获得新 Harness 的记录资格。

共享层现增加独立路径／时间参数化指令／实测刚体自碰撞检查，以及受约束的笛卡尔接近路径。
G2 叠碗的候选选择与有界跟踪补偿配置见 [任务准备流程](../../docs/harness/task_preparation.md)。
控制周期、夹爪开闭值和物理任务成功阈值保持原合同。
