# G2 bin 碰撞回退专项验收（2026-09-27）

任务配置已迁移到 [data_engine](../../data_engine/README.md)，保持配置内容与现有 HF
快照的字节一致。任务代码、模型和旧运行记录没有合并或复制到配置目录。

## 原因与修改

HF 版本 `082b4f1164ddd47ef5f0e775bdfeb14401691575` 中 bin 使用 convexDecomposition，
`maxConvexHulls=256`、`errorPercentage=1`、`shrinkWrap=true`。强制烘焙复现了
`GPU-compatible convex hull could not be built because of oblong shape` 及
`collision detection will fall back to CPU`，对应 `Bin_B04_01`。

仅增大 minThickness 没有修复问题。关闭 shrinkWrap、保留默认最小厚度 0.001 asset units 后通过。
修改在 G2 分类场景的 per-prim spawn physics 中生效；保留原视觉网格、资产尺寸和开口，
没有用单一凸包封住容器，没有修改 HF 资产字节或发布新 HF 版本。
代码使用 `isaaclab_arena/assets/convex_decomposition.py`，记录时保存其源码快照。

## 验证及证据

证据根目录：`outputs/harness/bin_collision_20260927/`（gitignored）。

- 原版对照：`source_verbose/report.json`，物理接触通过，但检测到 4 条 CPU 回退警告，整体 passed=false。
- 修复版：`qualified_verified/report.json`，警告捕获自检通过，0 条 CPU 回退警告，passed=true。
- 使用真实场景的 bin spawn 配置和 `[1.0, 1.15, 0.5]` 尺度，执行 1020 个 GPU 物理步。
  三处落物均在底部稳定，四方向速度冲击均被容器约束；底面相对原测量值最大偏差约 0.78 mm。
- 完整 G2 场景：`scene_verified.log` 与 `scene_smoke.py`。蓝、黄、绿三个 bin 分别触发烘焙检查，
  GPU dynamics=true、broadphase=GPU，完成 reset 和控制步，未检测到 CPU collision fallback。
- 18 项 Harness/标注测试与 36 项 G2 CLI、资产打包、运动/数据对齐回归测试通过。
- 两个家族的资产 preflight 通过。网页 8090/8091 HTTP 200，原 Pine 页面保留 20 个任务。

**警告捕获必须验证。** 普通初始化可能复用碰撞缓存；Kit 的静默启动还可能隐藏控制台警告。
检查工具启用 `--info`，设置警告级别并检查已知日志标记，触发独立烘焙路径。原始资产对照
用于验证检测确实有效。PhysX validator 曾在输出 CPU 回退警告的同时返回 compatibility=true，
因此该布尔值只能保留为诊断信息，不能单独授予 GPU 资格。
G2 collect/render 子进程现在也带 `--info`，使阶段日志保留 PhysX 警告。

这不是整条分类任务的成功/稳定性验收，也没有证明 native cuMotion 全部求解运行在 CUDA。
紧间隙新任务仍需验证实际 cooked 接触表面与规划用原始碰撞网格之间的误差。
旧资产、旧代码产生的回放应使用对应版本，不用新碰撞配置默默重处理旧数据。

## 重复验证

先按 [usage](usage.md) 设置 native venv 的 PYTHONPATH、EULA 与资产根目录，然后运行：

```bash
export ARENA_USDCRAFT_SCENE_ROOT="$PWD/local_assets/releases/082b4f1164ddd47ef5f0e775bdfeb14401691575"
.venv/bin/python tools/data_collection/check_g2_bin.py \
  --variant source --output outputs/harness/bin_source_new
.venv/bin/python tools/data_collection/check_g2_bin.py \
  --variant qualified --output outputs/harness/bin_qualified_new
```

每次使用新目录。source 是故障对照，允许输出 passed=false；qualified 检测到回退、警告捕获失败、
物体穿透或接触检查失败都会返回非零状态。几何烘焙准备与运行时 GPU 碰撞解算是不同阶段，
这里不宣称资产烘焙本身全部使用 CUDA。
