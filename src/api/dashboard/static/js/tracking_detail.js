let lastReidEvent = null;
let reidTimeline = [];

function setText(id, value) {
    const element = document.getElementById(id);
    if (element) element.textContent = value;
}

function getEventClass(level) {
    if (level === "WARNING") return "timeline-warning";
    if (level === "SUCCESS") return "timeline-success";
    return "timeline-info";
}

function setModeChipStyle(mode) {
    const chip = document.getElementById("currentModeValue");
    if (!chip) return;
    chip.textContent = mode;
    chip.classList.remove("mode-zone", "mode-class", "mode-marker", "mode-recovery");
    if (mode === "ZONE_TRACKING") chip.classList.add("mode-zone");
    else if (mode === "CLASS_TRACKING") chip.classList.add("mode-class");
    else if (mode === "MARKER_TRACKING") chip.classList.add("mode-marker");
    else chip.classList.add("mode-recovery");
}

function addTimeline(state, event, score, level) {
    const now = new Date().toLocaleTimeString();
    reidTimeline.unshift({ now, state, event, score, level });
    if (reidTimeline.length > 30) reidTimeline.pop();
    const box = document.getElementById("reidTimelineBox");
    if (!box) return;
    box.innerHTML = "";
    reidTimeline.forEach((item) => {
        const p = document.createElement("p");
        p.className = `timeline-item ${getEventClass(item.level)}`;
        p.textContent = `[${item.now}] ${item.state} / ${item.event} / score=${item.score}`;
        box.appendChild(p);
    });
}

async function fetchDetail() {
    const response = await fetch("/api/status");
    const data = await response.json();

    if (data.tracking_mode) {
        setModeChipStyle(data.tracking_mode.current_mode);
        setText("cameraModeValue", data.tracking_mode.camera_mode);
        setText("trackingSourceValue", data.tracking_mode.tracking_source);
        setText("targetTypeValue", data.tracking_mode.target_type);
        setText("arucoStatusValue", data.tracking_mode.aruco_status);
        setText("modeReasonValue", data.tracking_mode.mode_reason);
    }

    if (data.recovery) {
        setText("mismatchCountValue", data.recovery.mismatch_count);
        setText("recoveryTriggerValue", data.recovery.recovery_trigger);
        setText("recoveryActionValue", data.recovery.recovery_action);
        setText("recoveryStateValue", data.recovery.recovery_state);
        setText("lastRecoveryValue", data.recovery.last_recovery);
    }

    if (data.reid && lastReidEvent !== data.reid.event) {
        addTimeline(data.reid.state, data.reid.event, data.reid.score, data.reid.event_level);
        lastReidEvent = data.reid.event;
    }
}

fetchDetail();
setInterval(fetchDetail, 1000);
