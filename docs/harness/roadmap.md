# 分阶段实施与验收

这是整体实施顺序；当前开发范围及进度见 [development.md](development.md)，架构见 [architecture.md](architecture.md)。

## M0：整合基线与现有内容

- 将本地标注工作形成可保留的变更单元后整合已 fetch 的 data_engine；Git 提交/推送按后续授权执行。
- 合并 dataset.py 两方合同，保留 Pine/RR/G2 的任务家族、环境入口、校准和输出身份。
- 新 HF release 用单独目录下载验证，检查闭包、旧 Pine payload 与新 G2 entry；不覆盖活动版本。
- 输出 task/scenario/asset/profile 映射与 supported/unverified/unsupported 状态矩阵。
- 验收：数据/视频不进 Git；三套配置能解析；本地标注完整；旧预览链接和场景布局不变。

## M1：先冻结最小合同，再实现单 worker 核心

- 定义 AssetRef、Embodiment/Scene/Task/Scenario、RunSpec、EpisodeManifest、QualityReport 的
  最小版本化 schema。schema 只做数据描述，不 eager import simulator。
- 解析配置引用、能力依赖、动作/传感器 profile；明确时间合同、actor ID 和 provenance。
- 实现唯一受记录 step 通道、pre/post/terminal hooks、abort 与原子封存，复用现有 Arena recorder。
- 先接 Pine T001 与 G2 叠碗，检验 7D/16D、单/双臂与不同相机；此时不追求所有任务迁移。
- 验收：无硬编码 7/16/15fps/三相机的核心逻辑；preflight/list 不启动仿真；重复 hooks、提前 reset、
  空回合、丢帧、不完整转移、坏资产引用、错误动作空间均有明确失败结果。

## M2：采集、回放、标注、导出的闭环

- 将 T001/T041 skill trace 接到公共 hooks；G2 叠碗/插销 recipe 补同语义边界。
- 实现 live pre-action capture 的真实同步检查；保留 post-step preview；接入 bounded replay。
- 抽离 exporter 通用部分，保留机器人字段映射；标注和事件在导出/裁剪后有源区间映射。
- 分别保存在线任务判定与离线 raw 审计；合法失败和数据损坏采用不同保留/导出规则。
- 验收：同一 raw 可生成训练格式与网页预览；数据哈希、单位、时间、视频全解码、技能区间一致。
  错视频即使帧数正确也必须拒绝；重跑导出不可覆盖已有产物；标签修订不改变 raw。

## M3：资格流程与统一审核界面

- 从 Pine 复用“每任务一次成功后停下”的习惯，建立版本绑定的预览/审核/稳定性状态。
- 网页由统一 manifest 展示多机器人、多场景、多相机、阶段、物理/设备质量和失败原因。
- 定义宽范围工作区随机化与实际采样记录；加入关键阶段遮挡、抽屉扫掠、载物与相机碰撞检查。
- 明确 CUDA 验证证据，修复并单独验证 G2 CPU 碰撞回退；未知 planner device 保持 blocked。
- 验收：单次预览不会自动进入稳定性或批量；审核版本失效规则可复现；随机化确实改变布局。

## M4：代表任务跨家族回归

| 代表 case | 重点检验 |
|---|---|
| Pine T001 | 单臂 pick/place、7D、4 路审核流、skill/stage |
| Pine T041 | 关节对象状态、把手接触、开度审计、全程扫掠范围 |
| G2 stack_bowls | 两臂 actor/channel、不同动作布局、承载碰撞、重复 skill 实例 |
| G2 peg_into_sleeve | 精密接触、姿态/插入容差、回放误差是否可接受 |
| G2 clean_workcell_table | 多物体角色绑定、顺序任务、资产/碰撞 device 审计 |
| RR 一个抽屉或按压 case | 旧 recorder/导出合同兼容、资产方法身份、旧数据不被改写 |

每项先一个成功 pilot → 用户审核。审核之后才运行约定的随机化稳定性测试，报告总尝试数、成功数、
失败类型、采样拒绝率和场景覆盖，不只报告成功文件数。不用一个任务的通过替代其他任务资格。

## M5：批量调度与其余任务迁移

- 先单机 GPU lease、阶段预算、超时/取消、失败隔离、确定重试规则、断点从封存产物恢复。
- 性能报告分开启动/规划/物理/采集/编码/回放/导出，测量后再决定是否复用长驻环境或并发。
- 分阶段迁移 Pine 其余任务和 RR case；保留旧入口直到新路径对应验证完成。
- 组合数据集增加去重、split、过滤、混合权重和 action/schema 兼容检查；训练集发布与资产发布分开。
- 验收：中断恢复不重采已合格 episode、不重复计数、不同任务不会互相覆盖；批次能说明每个失败去向。

## M6：按需求增加更复杂能力

- 向量环境、移动操作、多机器人协作、灵巧手、柔性体、动态设施、异频力/触觉/深度传感器。
- policy/teleop 混合、人工纠正、恢复技能、课程和主动采样；每种来源记录其执行与标注 provenance。
- 分布式 worker、远程对象存储和集群调度通过 runtime/storage 接口扩展，不改变任务定义。
- 新后端/新模态/新训练格式进入独立适配器与 capability tests，未经验证不在 UI 标为 supported。

## 新内容如何接入

| 新增内容 | 提交的最小内容 | 首次验收 |
|---|---|---|
| 机器人/工具 | 资产引用、embodiment factory、控制/关节/传感器 profile、规划适配与 capability | 关节/FK/夹爪/碰撞/相机/记录 + 一个代表任务 |
| 场景 | scene factory/profile、工作区/设施/固定相机、一个 scenario | 尺度/布局/视野/扫掠范围/标定 + 一个任务预览 |
| 场景中的任务 | task 角色/判据、scenario 绑定、recipe 或其他动作来源、离线审计 | 一次成功、失败判据反例、标注/视频审核 |
| 任务资产变体 | 版本化 payload、交互接口/尺寸/碰撞描述、角色兼容与采样范围 | 依赖闭包、物理/可达性/接触 + 重新取得组合资格 |
| 传感器 | modality/schema、frame/time 定义、采样/capture/export adapter | 实际频率、同步/有效掩码、标定、缺帧反例 |
| 数据格式 | exporter 与字段/变换映射、版本和来源保存 | 同一 episode round-trip/loader/全量模态检查 |
| 动作来源/后端 | action/time/device/capability contract 与生命周期 adapter | reset/step/terminal/cancel/失败隔离及代表任务 |

## 测试与测量范围

纯 Python 合同、状态机、引用与幂等测试优先，不为薄配置重复写镜像测试。
仿真测试聚焦控制时序、自动 reset、传感器同步、物理判据和设备验证。
反例至少涵盖错误 joint order、夹爪单位、旧标定、坏视频来源、遗漏关节对象状态和标注错位。
性能必须用本机完整链路实测；远端单条耗时只作参考，不承诺批量耗时或跨机器确定性。

完成标准：用户能知道某个 scenario 是否可采集、为何不可采集、具体用了什么配置与设备、
一条数据怎样对应物理执行及技能阶段，以及增加新内容应该实现哪个接口。
