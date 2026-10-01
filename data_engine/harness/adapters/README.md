# Arena 采集器适配

此目录连接已有 Arena 任务家族，不注册新的机器人、资产或环境类。

- `catalogs.py`：读取家族已有任务配置，声明 scenario 引用、动作/图像合同、资产 ID、
  可执行能力和源文件依赖。扩展任务优先修改其家族配置，不能把模型或任务代码复制进这里。
- `existing.py`：把解析后的 spec 转成 Python 参数列表，调用原生入口；不用 shell 拼接，
  不覆盖输出目录，不额外修改动作或推进物理时间。
- `evidence.py`：将家族结果映射到公共质量报告，独立检查 raw、任务证据、图像/标签和来源。

增加一个新的采集家族时，在这些适配职责下新增模块/分派，不改公共 runner、锁、封存或状态机。
首版是仓库内显式适配，不提供任意外部插件自动发现，也不替代 Arena 自己的 registry。

每个 scenario 至少明确：现有 environment/embodiment 名、场景与任务 ID、对象绑定、
required_assets、action_profile、capture_profile、annotation_supported、preview_supported。
`source_patterns` 要覆盖该采集器、机器人/场景实现和配置；静态检查不能导入 simulator。

每个新适配先通过配置/错误路径测试，再做单次实际 preview。不能仅把 preview_supported 改为 true
就宣称已经取得资格。反例包括动作通道错误、旧资产、时间错位和错误视频来源。
当前只将已存在的执行路径接到数据管线，尚未自动支持新机器人控制方式或未知模态。
