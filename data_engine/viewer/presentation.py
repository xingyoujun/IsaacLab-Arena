# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Task grouping and readable labels over immutable collection evidence."""

import math

TITLES = {
    "g2/stack_bowls": "叠放三个碗",
    "g2/peg_into_sleeve": "将插销插入套筒",
    "g2/clean_workcell_table": "清理工作台",
    "pine_wm/T001": "将积木放入托盘",
    "pine_wm/T041": "拉开抽屉",
    "interactions/kettle_release": "热水壶按钮释放壶盖",
    "interactions/toaster_cancel": "烤面包机锁定与取消弹回",
}
STAGES = {
    "prepare_gripper": "准备夹爪",
    "select_grasp": "选择抓取",
    "approach": "接近物体",
    "descend": "下探",
    "align": "对准",
    "close": "闭合夹爪",
    "close_gripper": "闭合夹爪",
    "settle_grasp": "稳定抓取",
    "lift": "提起物体",
    "transfer": "搬运物体",
    "transport": "搬运物体",
    "lower": "下降放置",
    "settle_object": "等待物体稳定",
    "place": "放置物体",
    "actuate": "拉动抽屉",
    "insert": "插入",
    "release": "松开物体",
    "retreat": "撤离",
    "open_gripper": "张开夹爪",
    "verify": "验证结果",
    "verify_joint": "验证开度",
    "verify_placement": "验证放置",
    "initial_hold": "检查初始锁扣状态",
    "return_to_ready": "返回准备姿态",
    "configure_finger_press_lid_release": "准备按壶盖按钮",
    "approach_press_lid_release": "接近壶盖按钮",
    "contact_press_lid_release": "对准壶盖按钮",
    "actuate_press_lid_release": "按下壶盖按钮",
    "retreat_press_lid_release": "撤离壶盖按钮",
    "hold_press_lid_release": "确认壶盖弹开",
    "configure_finger_lower_bread": "准备压下托架",
    "approach_lower_bread": "接近托架压柄",
    "contact_lower_bread": "对准托架压柄",
    "actuate_lower_bread": "压下托架",
    "retreat_lower_bread": "撤离托架压柄",
    "hold_lower_bread": "确认托架锁住",
    "configure_finger_press_cancel": "准备按取消按钮",
    "approach_press_cancel": "接近取消按钮",
    "contact_press_cancel": "对准取消按钮",
    "actuate_press_cancel": "按下取消按钮",
    "retreat_press_cancel": "撤离取消按钮",
    "hold_press_cancel": "确认托架弹回",
}
SKILLS = {
    "verify_initial_latch": "检查机构初态",
    "prepare_contact_tool": "准备夹爪",
    "press": "按压并检查联动",
    "transit": "切换操作位置",
    "open_articulated": "抓住把手并拉开抽屉",
    "pick_place": "抓取并放置",
    "pick": "抓取积木",
    "place": "将积木放入托盘",
}


def default_name(record, scenario):
    """Describe recorded configuration without claiming new qualification."""
    if record["scenario"] == "pine_wm/T041":
        layout = scenario.get("layout", {})
        if layout.get("layout_contract"):
            return "D435 正面布局 · 中心区域抽拉"
        return "旧布局 · 抽屉朝向第三视角"
    if record["scenario"] == "g2/stack_bowls":
        if scenario.get("task_plan", {}).get("schema") == "arena.task_plan.v1":
            return "外侧抓碗 · 独立安全检查"
        return "初版抓碗路径"
    return {
        "pine_wm/T001": "积木入托盘 · 基线预览",
        "g2/peg_into_sleeve": "插销入套筒 · 基线验证",
        "g2/clean_workcell_table": "三件物体分类入箱",
    }.get(record["scenario"], "任务预览")


def timeline(record):
    """Validate recorded intervals, excluding zero-duration planning events from playback."""
    count = record["quality"].get("raw", {}).get("steps", 0)
    dt = record.get("step_dt", 0)
    if not isinstance(dt, (int, float)) or not math.isfinite(dt) or dt <= 0 or not count:
        return []
    result = []
    previous = 0
    skills = {skill["id"]: skill for skill in record.get("skills", [])}
    for stage in record["stages"]:
        start, end = stage.get("start_step"), stage.get("end_step")
        if not isinstance(start, int) or not isinstance(end, int) or not 0 <= start <= end <= count:
            return []
        if start == end:
            continue
        if start < previous:
            return []
        previous = end
        name = stage["name"]
        verb = name.split(": ")[-1].split(" ")[0]
        label = STAGES.get(verb, name)
        skill = skills.get(stage.get("skill_instance_id"))
        group = record["task_title"]
        if skill:
            group = SKILLS.get(skill["name"], skill["name"])
        elif record["scenario"] == "g2/stack_bowls":
            if name.startswith("right:"):
                group = "右手：右碗叠到中间碗"
            elif name.startswith("left:"):
                group = "左手：左碗叠到碗塔"
            else:
                group = "检查碗塔"
        result.append(
            dict(
                label=label, group=group, original=name, start=start * dt, end=end * dt, start_step=start, end_step=end
            )
        )
    return result


def group_tasks(records):
    """Show one entry per task with ordered attempts and a usable default preview."""
    groups = {}
    for record in sorted(records, key=lambda r: (r["created_at"], r["name"])):
        key = record["scenario"]
        group = groups.setdefault(
            key, dict(id=key, title=record["task_title"], embodiment=key.split("/")[0], attempts=[])
        )
        group["attempts"].append(record)
        record["attempt_number"] = len(group["attempts"])
    for group in groups.values():
        attempts = group["attempts"]
        usable = [r for r in attempts if r["videos"] and r["diagnostic_review"].get("status") != "blocked"]
        videos = [r for r in attempts if r["videos"]]
        group["default_attempt"] = (usable or videos or attempts)[-1]["name"]
        group["attempts"] = list(reversed(attempts))
    return sorted(groups.values(), key=lambda group: group["id"])
