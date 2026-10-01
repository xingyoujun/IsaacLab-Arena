# 通用 Skill / Sub-stage 标注

`arena.skill_trace.v1` 描述执行轨迹的语义阶段，与机器人型号、关节数量、控制器和规划器无关。
`SkillTrace` 只依赖 Python 标准库；HDF5 导出函数按需加载 h5py / NumPy。
当前只在 Pine WM T001、T041 做接入试验，其他机器人和任务尚未自动接入。

面向全部采集链路的组合、时钟和扩展边界见 [Harness 架构](../../docs/harness/README.md)。
其中多主体/异频流的接口属于后续设计，不改变当前 v1 的单 stream 语义。

## 层级和身份

- Episode：`task_id`、`env_id`、`actor_id`、`source`、最终任务结果。
- Skill instance：本回合内唯一整数 `id`，语义名称、对象/目标、参数、起止控制步。
- Stage instance：本回合内唯一整数 `id`，所属 `skill_instance_id`、阶段名称、起止步。
- Event：某个控制边界上的瞬时事实，例如候选被拒绝、物体抬升验证、抽屉开度验证。

语义名称可以重复，instance ID 不重复。例如五次 `pick` 分别绑定五个物体。
`entities` 使用任务中的稳定对象 ID，而非强制要求 USD prim 路径。
`actor_id` 可以表示单臂、左右臂、夹爪或移动底盘，不强制某种 embodiment。
每个环境/执行主体维护自己的 trace；并行主体需由接入端保持相同控制时钟，再分别导出。
当前实现不把不同频率的主体自动重采样，也不声称已完成向量环境接入。

## 公共词汇与结果

| Skill | 常见 sub-stage |
|---|---|
| `pick` | `prepare_gripper`, `approach`, `align`, `close_gripper`, `settle_grasp`, `lift` |
| `place` | `transport`, `lower`, `release`, `settle_object`, `retreat`, `verify_placement` |
| `open_articulated` | `prepare_gripper`, `approach`, `align`, `close_gripper`, `settle_grasp`, `actuate`, `release`, `retreat`, `open_gripper`, `verify_joint` |

任务接入可扩展词汇（例如 `push`、`navigate`），无需修改底层记录器。
对象名称和目标参数放实体/参数字段，不拼进 skill 名称。词汇意义改变时应更新 schema/适配版本。

`execution_status=completed` 仅代表这段代码执行结束；不等于抓取或放置成功。
Skill 的 `success` 默认 `null`，设置 true/false 必须附带实测 `evidence`。
最终任务成功不会自动覆盖所有子阶段的结果。碰撞检查仍使用原来的独立 `phase`，不读取标注字段。

## 时间契约

区间为 `[start_step, end_step)`，索引对齐完成的控制转移：

```text
observation[t] --action[t], skill/stage[t]--> post_state[t]
                                                |
                                     live preview frame[t]
```

HDF5 observation 是步前状态；当前 Pine WM 视频是步后状态，仿真时刻为 `(t+1)*step_dt`。
视频从第一个步后图像开始，播放器跳转到对应帧使用 `t*step_dt`。
不能把预览 frame[t] 当作训练用 observation[t] 图像；训练图像需遵循单独的时间对齐契约。
标注基于控制步，不使用墙钟时间，因此规划耗时不会变成额外视频帧。

接入顺序：在 env.step 前调用 `before_step()` 冻结标签，返回后调用 `after_step()` 计一个完成步。
执行中不允许切换阶段。步骤抛异常时调用 `abort_step()`，记录不完整转移而不伪造完整样本。
若失败 HDF5 已写入部分转移导致长度不一致，只保留诊断 sidecar，不发布对齐标注。

零长度阶段用于表示未执行动作的规划拒绝；它们没有帧，但保留在阶段表和事件中。
实际已执行过的失败接近路径也保留，不因后续重试成功而删除。

## 最小接入示例

```python
trace = SkillTrace(task_id="my_task", step_dt=env.step_dt, actor_id="left_arm")
trace.start_skill("pick", entities={"object": "cube"})
trace.stage("approach")
trace.before_step()
env.step(action)
trace.after_step()
# 在控制器语义边界切换阶段，而不是根据视频运动幅度猜测。
trace.end_skill()  # 没有物理证据时，success 保持未知。
document = trace.finish(success=task_success)
write_hdf5_trace(demo_group, document, expected_steps=num_actions)
```

初始化/复位过程应在创建 trace 前完成；每次 reset 创建新的实例。
`source` 记录标注来源（如 `script`、`human`、`inferred`）；未来遥操作或策略接入可以使用同一格式，
但必须自行产生可信的阶段边界，记录器不会自动推断意图。

## 输出和验收

每回合一个 JSON sidecar；HDF5 同时包含：

```text
annotations/metadata_json
annotations/step_skill_ids       # N 个 skill instance ID
annotations/step_stage_ids       # N 个 stage instance ID
```

检查阶段层级、起止边界、逐动作覆盖和长度一致性；与动作或视频帧数不一致时拒绝成功标注。
Pine WM 网页只展示通过 HDF5 对齐校验、且匹配当前视频的标注，可按阶段跳转。

扩展顺序：先确认 T001/T041 的语义与边界，再接入其他 Pine WM 任务，最后为其他机器人任务增加
轻量适配器。RR 与其他项目无需复制 Pine WM 任务代码，只复用这个公共模块及数据格式。

2026-09-26 首轮试验：T001、T041 各成功一回合，分别记录 419、280 个控制步；
每回合四路 15 fps 视频。T001 包含 `pick` / `place`，T041 包含 `open_articulated`。
结果位于实验目录 `review_v3_skills/<task>`，大文件不进 Git。完成后已关闭标注试跑开关，
等待阶段划分审核；本次不代表稳定性测试通过。
