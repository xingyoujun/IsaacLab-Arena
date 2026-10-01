/* Task dashboard. All labels from recordings are rendered as text, never as HTML. */
(() => {
    const $ = (s, el = document) => el.querySelector(s);
    let tasks = JSON.parse($("#initial-data").textContent);
    let selected = null,
        selectedTask = null,
        activeVideo = null,
        refreshBusy = false;
    let renderedSignature = "",
        cameraName = null,
        frameLoop = 0;
    const cameraLabels = {
        head: "头部相机",
        left_wrist: "左腕",
        right_wrist: "右腕",
        realsense_d435: "D435 · 训练主视角",
        wrist_a: "腕部 A",
        wrist_b: "腕部 B",
        scene_cam: "第三视角 · 审核",
    };
    const node = (tag, cls = "", text = null) => {
        const e = document.createElement(tag);
        if (cls) e.className = cls;
        if (text !== null) e.textContent = text;
        return e;
    };
    const button = (text, cls, action) => {
        const b = node("button", cls, text);
        b.type = "button";
        b.addEventListener("click", action);
        return b;
    };
    const allRuns = () => tasks.flatMap((t) => t.attempts);
    const current = () => allRuns().find((r) => r.name === selected);
    const titleStatus = (r) => {
        if (r.diagnostic_review?.status === "blocked")
            return ["审核阻断", "blocked"];
        if (
            ["collection_failed", "audit_failed", "cancelled"].includes(r.state)
        )
            return ["尝试失败", "failed"];
        if (r.state === "preview_validated") return ["预览通过 · 待审核", "ok"];
        if (r.state === "collecting") return ["正在执行", ""];
        return ["等待结果", ""];
    };
    const badge = (r) => {
        const [text, cls] = titleStatus(r);
        return node("span", "badge " + cls, text);
    };
    const formatTime = (n) =>
        `${Math.floor(n / 60)}:${String(Math.floor(n % 60)).padStart(2, "0")}`;
    const date = (iso) =>
        new Date(iso).toLocaleString("zh-CN", {
            month: "2-digit",
            day: "2-digit",
            hour: "2-digit",
            minute: "2-digit",
            timeZone: "UTC",
        }) + " UTC";
    function toast(text) {
        const el = $("#toast");
        el.textContent = text;
        el.hidden = false;
        clearTimeout(toast.timer);
        toast.timer = setTimeout(() => (el.hidden = true), 3500);
    }
    function sidebar() {
        const list = $("#tasks");
        list.replaceChildren();
        const query = $("#search").value.toLowerCase();
        $("#counts").textContent =
            `${tasks.length} 个 · ${allRuns().length} 次尝试`;
        tasks
            .filter((t) => (t.title + " " + t.id).toLowerCase().includes(query))
            .forEach((t) => {
                const b = button(
                    "",
                    "task" + (t.id === selectedTask ? " active" : ""),
                    () => choose(t.default_attempt),
                );
                b.append(
                    node("span", "robot", t.embodiment.toUpperCase()),
                    node("strong", "", t.title),
                );
                const meta = node("span", "meta");
                meta.append(
                    node("span", "", t.id.split("/")[1]),
                    node("span", "", `${t.attempts.length} 次尝试`),
                );
                b.append(meta);
                list.append(b);
            });
    }
    function choose(name) {
        const r = allRuns().find((r) => r.name === name);
        if (!r) return;
        if (selected !== name) {
            if (activeVideo) activeVideo.pause();
            cameraName = null;
        }
        selected = name;
        selectedTask = r.scenario;
        history.replaceState(null, "", "#" + encodeURIComponent(name));
        sidebar();
        render(true);
    }
    function renameForm(r, holder) {
        if (holder.querySelector("form")) return;
        const form = node("form", "rename");
        const input = node("input");
        input.value = r.display_name;
        input.maxLength = 80;
        input.required = true;
        input.setAttribute("aria-label", "尝试名称");
        const save = node("button", "secondary", "保存名称");
        save.type = "submit";
        form.append(
            input,
            save,
            button("取消", "secondary", () => form.remove()),
        );
        holder.append(form);
        input.focus();
        input.select();
        form.addEventListener("submit", async (e) => {
            e.preventDefault();
            save.disabled = true;
            try {
                const response = await fetch("/api/attempt-name", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        run: r.name,
                        spec_id: r.spec_id,
                        display_name: input.value,
                    }),
                });
                if (!response.ok)
                    throw new Error("保存失败，请检查名称或刷新重试");
                r.display_name = input.value.trim();
                $(".display-name", holder).textContent = r.display_name;
                form.remove();
                renderedSignature = JSON.stringify(r);
                renderHistory();
                toast("名称已保存；原始记录保持不变");
            } catch (error) {
                toast(error.message);
                save.disabled = false;
            }
        });
    }
    function renderHistory() {
        const holder = $("#attempt-history");
        if (!holder) return;
        const task = tasks.find((t) => t.id === selectedTask);
        $("summary", holder).textContent =
            `尝试历史 · ${task.attempts.length} 次（展开切换 / 命名）`;
        const list = $(".history-list", holder);
        list.replaceChildren();
        task.attempts.forEach((r) => {
            const row = button(
                "",
                "history-row" + (r.name === selected ? " selected" : ""),
                () => choose(r.name),
            );
            const text = node("div");
            text.append(
                node(
                    "strong",
                    "",
                    `第 ${r.attempt_number} 次 · ${r.display_name}`,
                ),
                node("small", "", date(r.created_at)),
            );
            row.append(text, badge(r));
            list.append(row);
        });
    }
    function details(title, data) {
        const el = node("details");
        el.append(node("summary", "", title));
        const body = node("div", "details-body");
        body.append(node("pre", "", JSON.stringify(data, null, 2)));
        el.append(body);
        return el;
    }
    function render(force = false) {
        const r = current();
        if (!r) return;
        const signature = JSON.stringify(r);
        if (!force && signature === renderedSignature) {
            renderHistory();
            return;
        }
        const savedTime = activeVideo?.currentTime || 0,
            wasPlaying = activeVideo && !activeVideo.paused;
        const historyOpen = $("#attempt-history")?.open || false;
        if (activeVideo) activeVideo.pause();
        activeVideo = null;
        frameLoop++;
        renderedSignature = signature;
        const main = $("#workspace");
        main.replaceChildren();
        const task = tasks.find((t) => t.id === r.scenario);
        const heading = node("div", "task-heading");
        const titles = node("div");
        titles.append(
            node("span", "eyebrow", task.embodiment.toUpperCase()),
            node("h2", "", task.title),
            node("div", "muted", task.id),
        );
        heading.append(titles, badge(r));
        main.append(heading);
        const panel = node("section", "panel");
        const top = node("div", "attempt-top");
        const names = node("div", "attempt-title");
        names.append(
            node("span", "number", `第 ${r.attempt_number} 次尝试`),
            node("h3", "display-name", r.display_name),
        );
        top.append(
            names,
            button("修改名称", "secondary", () => renameForm(r, top)),
        );
        panel.append(top);
        const summary = r.diagnostic_review?.summary;
        if (summary) panel.append(node("p", "warning", summary));
        if (r.videos.length)
            buildPlayer(panel, r, !force ? savedTime : 0, !force && wasPlaying);
        else
            panel.append(
                node(
                    "div",
                    "empty-video",
                    r.state === "collecting"
                        ? "此尝试正在执行，完成后视频会自动出现在这里。"
                        : "此次尝试没有已验收视频。失败与检查记录保留在下方。",
                ),
            );
        main.append(panel);
        const historyBox = node("details");
        historyBox.id = "attempt-history";
        historyBox.open = historyOpen;
        historyBox.append(node("summary"), node("div", "history-list"));
        main.append(historyBox);
        renderHistory();
        if (r.planning_events?.length)
            main.append(
                details(
                    `规划事件 · ${r.planning_events.length} 项（不占视频时长）`,
                    r.planning_events,
                ),
            );
        main.append(
            details("任务拆分与抓取选择", {
                任务配置: r.task_plan,
                抓取候选: r.decisions.filter(
                    (d) => d.kind === "grasp_candidate",
                ),
                阶段来源: r.stage_source || "此记录没有阶段标注",
            }),
        );
        main.append(
            details("质量与原始记录", {
                原始目录: r.name,
                时间: r.created_at,
                失败: r.failure,
                审核: r.diagnostic_review,
                质量: r.quality,
                事件: r.events,
            }),
        );
        const links = node("p", "footer-note");
        const a = node("a", "", "原始 manifest");
        a.href = "/" + encodeURIComponent(r.name) + "/run.json";
        a.target = "_blank";
        links.append(
            a,
            document.createTextNode(
                " · 预览通过不等于训练就绪；稳定性与批量采集未开放。",
            ),
        );
        main.append(links);
    }
    function buildPlayer(panel, r, initialTime, resume) {
        const layout = node("div", "player-layout"),
            area = node("div", "video-area"),
            video = node("video");
        video.controls = true;
        video.preload = "metadata";
        video.playsInline = true;
        activeVideo = video;
        const camera =
            r.videos.find((v) => v.name === cameraName) || r.videos[0];
        cameraName = camera.name;
        area.append(video);
        const cameras = node("div", "cameras");
        function setCamera(v, time, play) {
            cameraName = v.name;
            video.src = v.url;
            video.addEventListener(
                "loadedmetadata",
                () => {
                    video.currentTime = Math.min(
                        time,
                        Math.max(0, video.duration - 0.001),
                    );
                    if (play) video.play().catch(() => {});
                    sync();
                },
                { once: true },
            );
            video.load();
            for (const b of cameras.children)
                b.classList.toggle("active", b.dataset.camera === cameraName);
        }
        r.videos.forEach((v) => {
            const b = button(cameraLabels[v.name] || v.name, "", () =>
                setCamera(v, video.currentTime, !video.paused),
            );
            b.dataset.camera = v.name;
            cameras.append(b);
        });
        area.append(cameras);
        layout.append(area);
        const steps = node("div", "stage-list");
        steps.append(node("h3", "", "子任务 / 动作阶段"));
        let lastGroup = null;
        const timeline = r.timeline || [];
        const stepButtons = [];
        timeline.forEach((s, i) => {
            if (s.group !== lastGroup) {
                steps.append(node("div", "skill-label", s.group));
                lastGroup = s.group;
            }
            const b = button("", "step", () => seek(s.start));
            b.append(
                node("span", "step-num", String(i + 1).padStart(2, "0")),
                node("span", "", s.label),
            );
            b.title = s.original;
            steps.append(b);
            stepButtons.push(b);
        });
        if (!timeline.length)
            steps.append(
                node(
                    "p",
                    "muted",
                    "此记录没有可对齐的阶段标注，仅显示整体播放进度。",
                ),
            );
        layout.append(steps);
        panel.append(layout);
        const playback = node("div", "playback");
        const now = node("div", "now");
        const currentLabel = node("div");
        currentLabel.append(node("div", "label", "当前执行"));
        const group = node("small"),
            stage = node("strong");
        currentLabel.append(group, stage);
        const time = node("div", "time");
        now.append(currentLabel, time);
        playback.append(now);
        const bar = node("div", "timeline");
        const segments = [];
        const total = (r.quality.raw?.steps || 0) * (r.step_dt || 1 / 15);
        timeline.forEach((s, i) => {
            const b = button("", "segment", () => seek(s.start));
            b.style.left = (s.start / total) * 100 + "%";
            b.style.width = ((s.end - s.start) / total) * 100 + "%";
            b.title = `${s.group} · ${s.label}`;
            b.setAttribute("aria-label", `跳转到${s.group}，${s.label}`);
            bar.append(b);
            segments.push(b);
        });
        const cursor = node("div", "playhead");
        bar.append(cursor);
        playback.append(bar);
        const scrub = node("input", "scrubber");
        scrub.type = "range";
        scrub.min = 0;
        scrub.max = 1000;
        scrub.step = 1;
        scrub.value = 0;
        scrub.setAttribute("aria-label", "视频与子任务进度");
        scrub.addEventListener("input", () =>
            seek(
                (Number(scrub.value) / 1000) * (video.duration || total),
                false,
            ),
        );
        playback.append(scrub);
        const legend = node("div", "legend");
        legend.append(
            node(
                "span",
                "",
                timeline.length
                    ? "点击动作阶段或拖动进度条，视频同步跳转"
                    : "阶段标注缺失 · 不推断子任务",
            ),
            node(
                "span",
                "",
                r.stage_source === "recorded_skill_trace"
                    ? "录制阶段标注"
                    : timeline.length
                      ? "采集器阶段记录"
                      : "",
            ),
        );
        playback.append(legend);
        panel.append(playback);
        let activeIndex = -2;
        function sync(mediaTime = video.currentTime) {
            if (typeof mediaTime !== "number") mediaTime = video.currentTime;
            const duration = Number.isFinite(video.duration)
                ? video.duration
                : total;
            const t = Math.min(mediaTime, duration);
            const progress = duration > 0 ? t / duration : 0;
            scrub.value = progress * 1000;
            cursor.style.left = Math.min(99.8, progress * 100) + "%";
            time.textContent = `${formatTime(t)} / ${formatTime(duration)} · ${Math.round(progress * 100)}%`;
            let index = timeline.findIndex(
                (s) => t >= s.start - 1e-6 && t < s.end - 1e-6,
            );
            if (
                video.ended &&
                timeline.length &&
                Math.abs(t - timeline.at(-1).end) < 0.1
            )
                index = timeline.length - 1;
            const s = timeline[index];
            group.textContent = s?.group || r.task_title;
            stage.textContent = s
                ? `${video.ended ? "播放结束 · " : ""}${s.label}`
                : "未记录此时刻的阶段";
            if (index !== activeIndex) {
                activeIndex = index;
                stepButtons.forEach((b, i) => {
                    b.classList.toggle("active", i === index);
                    b.classList.toggle("done", timeline[i].end <= t);
                    if (i === index) b.setAttribute("aria-current", "step");
                    else b.removeAttribute("aria-current");
                });
                segments.forEach((b, i) =>
                    b.classList.toggle("active", i === index),
                );
            }
        }
        function seek(t, play = true) {
            const move = () => {
                video.currentTime = Math.max(
                    0,
                    Math.min(t, video.duration - 0.001),
                );
                sync(t);
                if (play) video.play().catch(() => {});
            };
            if (video.readyState) move();
            else video.addEventListener("loadedmetadata", move, { once: true });
        }
        for (const event of [
            "timeupdate",
            "seeked",
            "loadedmetadata",
            "ended",
            "pause",
        ])
            video.addEventListener(event, () => sync());
        const token = frameLoop;
        function frame(now, meta) {
            if (token !== frameLoop) return;
            sync(meta.mediaTime);
            video.requestVideoFrameCallback(frame);
        }
        if (video.requestVideoFrameCallback)
            video.requestVideoFrameCallback(frame);
        setCamera(camera, initialTime, resume);
        sync(0);
    }
    async function refresh() {
        if (refreshBusy) return;
        refreshBusy = true;
        try {
            const response = await fetch("/api/tasks", { cache: "no-store" });
            if (!response.ok) throw new Error();
            tasks = await response.json();
            sidebar();
            if (current()) render();
            else if (tasks.length) choose(tasks[0].default_attempt);
            $("#sync-status").textContent =
                "自动同步 · " +
                new Date().toLocaleTimeString("zh-CN", { hour12: false });
        } catch {
            $("#sync-status").textContent = "连接中断 · 正在重试";
        } finally {
            refreshBusy = false;
        }
    }
    $("#search").addEventListener("input", sidebar);
    window.addEventListener("hashchange", () =>
        choose(decodeURIComponent(location.hash.slice(1))),
    );
    const initial = decodeURIComponent(location.hash.slice(1));
    if (allRuns().some((r) => r.name === initial)) choose(initial);
    else if (tasks.length) {
        const recent = allRuns()
            .filter(
                (r) =>
                    r.videos.length &&
                    r.diagnostic_review?.status !== "blocked",
            )
            .sort((a, b) => b.created_at.localeCompare(a.created_at))[0];
        choose(recent?.name || tasks[0].default_attempt);
    } else $("#empty").textContent = "尚无任务尝试；新任务运行后会自动出现。";
    refresh();
    setInterval(refresh, 10000);
})();
