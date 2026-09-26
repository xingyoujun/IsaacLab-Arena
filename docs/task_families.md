# RR real2sim 与 Pine WM：任务归属

两者共用 Arena 与 UR7e 基础设施，但任务、资产配置、标定和数据独立管理。
不能因使用同类机器人，或都具有“开抽屉”动作，就把两套任务互相复制。

| 项目 | RR real2sim | Pine WM |
|---|---|---|
| 场景入口 | `ur7e_workcell` 和 RR 各任务环境 | `pine_wm`、`pine_wm_first20` |
| 机器人 embodiment | `ur7e_robotiq_joint_pos` / `ur7e_robotiq_ik` | `pine_wm_ur7e` |
| 任务 | open_drawer、press_toaster、turn_toaster_knob；四种资产方法 | 首批 T001–T145 中的 20 项；200 项目录是规划清单 |
| 任务实现 | `ur7e_*` 环境及其 RR collection drivers | `pine_wm_first20_environment.py`、`tools/pine_wm/first20/` |
| 对象命名 | `usdcraft_*`、`articraft_*`、`miniworkflow_*`；保留既有录制键 | manifest 的 `P20_*`，场景实例键由 first20 配置决定 |
| 抽屉区别 | `drawer_rr`，球形把手的 RR USDcraft 抽屉 | `P20_drawer`，280 mm 横杆把手抽屉，T041–T045 共用 |
| 资产适配层 | `tools/rr_sim2real/asset_overlays/`，属于 RR | 实例缩放和标记在 Pine WM 环境代码中处理 |
| 运行模型 | 按 `docs/rr_sim2real/assets.json` 配置的外部 RR 资产 | `local_assets/USDCraft-Scene/`，HF 锁定版本 |
| 文档 | `docs/rr_sim2real/`、`tools/rr_sim2real/README.md` | `docs/pine_wm.md`、`docs/pine_wm_first20.md` |

## 可复用基础代码

UR7e 关节控制、记录器、黑色夹爪材质、相机工具函数、cuMotion 接口和渲染工具可以共用。
Pine WM 有自己的标定、安装支架、规划碰撞描述和任务成功判据。
`pine_wm_environment.py` 复用 workcell 的桌面块构造和 IdleTask 等小型工具，
这不意味着加载了 RR 的任务对象或数据集。

## RR 的七个 USDA

这些文件约 12 KB，是与 RR 环境一起维护的物理/材质适配配置，不是 Pine WM 模型。
本次保留其 Git 跟踪和路径；不因为 Pine WM 使用 HF 就迁移 RR 的资产体系。
其中 `boxx_support.usda` 被 RR 烤面包机旋钮任务及准备脚本直接引用，仍具有本机外部
`boxx.usdc` 依赖。它的跨机器迁移应在 RR 范围内处理，不通过 Pine WM 修补。

## 已退役的交叉验证

最初曾将 RR USDcraft 抽屉接入 Pine WM，环境名为 `pine_wm_usdcraft_open_drawer`。
这个早期验证入口现已移除，RR 抽屉驱动恢复 RR 默认配置。
当前 20 项任务及网页视频不依赖该入口，历史录制文件保留在外部实验目录。
第一版 HF 包中留存的 RR 抽屉是独立对象，不代表 RR 任务已迁入 Pine WM。
