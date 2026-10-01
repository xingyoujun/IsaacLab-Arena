# 仿真数据采集 Harness

状态：2026-09-26 开始实现开发版。完整架构是目标设计，实际可用范围以开发验收记录为准。
Harness 是 Arena 上的数据生成管线，复用环境/机器人/任务注册与执行接口，不另建一套 Arena。

目标是让机器人、场景、任务、资产和采集方式可以独立扩展，同时让每条数据的来源、时间含义、
质量及审核状态可以验证。统一的是运行与数据契约，不强制所有机器人使用相同动作维度、相机或规划器。

- [架构与扩展契约](architecture.md)：领域模型、底层选择、接口职责、数据与运行生命周期。
- [现有内容盘点](inventory.md)：本地、远端及实验成果分别有哪些；哪些复用、改造或暂缓。
- [实施与验收](roadmap.md)：分阶段迁移、回归矩阵、首次接入检查清单。
- [开发与验收记录](development.md)：开发路径、逐步检查和当前状态。
- [开发版命令](usage.md)：计划、预览、审计与产物说明。
- [首版验收标准](acceptance.md)：软件、任务与数据集的不同通过条件。

审计基线：本地 HEAD `ccefc86a4b0aed15bbc431959d3932fa50e1a8f1`，加本地未提交的
`data_engine/annotations/` 和 T001/T041 标注接入；远端
`fork/chuanruiz/data_engine` 为 `09a786fb43241fc1d3a402f2b07cbcb935e94ee0`。
开发启动后已 fast-forward 整合该远端提交，并保留本地标注。旧审计基线不代表当前工作树；
本机新运行证据另记于开发记录，远端报告不替代本机验收。

架构决策：采用 Arena 现有环境组合 + 环境控制步/recorder hooks 作为首个运行底层，
在其上建立不依赖机器人和规划器的 harness；从 G2、Pine WM、RR 分别吸收已验证能力。
G2 Session、Pine WM qualify 和 RR 批处理都作为迁移来源，不指定其中一个为所有场景的主循环。

已有任务家族边界仍见 [RR / Pine WM 说明](../task_families.md)。任务目录是能力清单，
注册、成功预览、稳定性通过、可正式采集是不同状态，不能混为一谈。

任务配置已统一到 [data_engine/](../../data_engine/README.md)。
G2 bin 的冷烘焙回退修复见 [2026-09-27 验收](bin_collision_20260927.md)。

当前源码总入口：[data_engine/README.md](../../data_engine/README.md)。
官方 main 对照：[改动与归属](upstream_diff_20260927.md)。
