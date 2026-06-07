console.log("Dashboard Loaded");

let lastTargetId = null;
let lastReidEvent = null;
let lastReidState = null;
let lastHealthLevel = null;
let lastTrackingMode = null;
let lastRecoveryState = null;
let lastConflictStatus = null;
let reidTimeline = [];

function setText(id, value) {
    const element = document.getElementById(id);
    if (element) {
        element.textContent = value;
    }
}

function addLog(message) {
    const logBox = document.getElementById("log-box");
    const p = document.createElement("p");
    const time = new Date().toLocaleTimeString();
    p.textContent = `[${time}] ${message}`;
    logBox.appendChild(p);
    logBox.scrollTop = logBox.scrollHeight;
}

function getEventClass(level) {
    if (level === "WARNING") return "timeline-warning";
    if (level === "SUCCESS") return "timeline-success";
    return "timeline-info";
}

function addReidTimelineEvent(state, event, score, level) {
    const now = new Date().toLocaleTimeString();
    reidTimeline.unshift({ time: now, state, event, score, level });

    if (reidTimeline.length > 6) {
        reidTimeline.pop();
    }

    renderReidTimeline();
}

function renderReidTimeline() {
    const box = document.getElementById("reidTimelineBox");
    if (!box) return;

    box.innerHTML = "";

    reidTimeline.forEach((item) => {
        const p = document.createElement("p");
        p.className = `timeline-item ${getEventClass(item.level)}`;
        p.textContent = `[${item.time}] ${item.state} / ${item.event} / score=${item.score}`;
        box.appendChild(p);
    });
}

function setHealthLevelStyle(level) {
    const badge = document.getElementById("healthLevelValue");
    if (!badge) return;

    badge.textContent = level;
    badge.classList.remove("health-good", "health-caution", "health-warning");

    if (level === "GOOD") {
        badge.classList.add("health-good");
    } else if (level === "CAUTION") {
        badge.classList.add("health-caution");
    } else {
        badge.classList.add("health-warning");
    }
}

function setModeChipStyle(mode) {
    const chip = document.getElementById("currentModeValue");
    if (!chip) return;

    chip.textContent = mode;
    chip.classList.remove("mode-zone", "mode-class", "mode-marker", "mode-recovery");

    if (mode === "ZONE_TRACKING") {
        chip.classList.add("mode-zone");
    } else if (mode === "CLASS_TRACKING") {
        chip.classList.add("mode-class");
    } else if (mode === "MARKER_TRACKING") {
        chip.classList.add("mode-marker");
    } else {
        chip.classList.add("mode-recovery");
    }
}

async function callAPI(url, message) {
    try {
        const response = await fetch(url, { method: "POST" });
        const data = await response.json();

        addLog(`${message} 성공`);
        console.log(data);

        fetchStatus();
        fetchDetections();
    } catch (error) {
        addLog(`${message} 실패`);
        console.error(error);
    }
}

async function fetchStatus() {
    try {
        const response = await fetch("/api/status");
        const data = await response.json();

        setText("detectorValue", data.detector);
        setText("trackerValue", data.tracker);
        setText("fpsValue", data.fps);
        setText("targetIdValue", data.target_id);
        setText("zoneLockValue", data.zone_lock);
        setText("ptzValue", data.ptz_status);
        setText("sessionValue", data.session_state);
        setText("sessionBadge", `Session ${data.session_state}`);

        if (data.health) {
            setText("healthScoreValue", data.health.score);
            setHealthLevelStyle(data.health.level);
            setText("healthTrackingValue", data.health.tracking);
            setText("healthReidValue", data.health.reid);
            setText("healthPtzValue", data.health.ptz);
            setText("healthWarningsValue", data.health.warnings);

            if (lastHealthLevel !== null && lastHealthLevel !== data.health.level) {
                addLog(`System Health changed: ${lastHealthLevel} → ${data.health.level}`);
            }

            lastHealthLevel = data.health.level;
        }

        if (data.tracking_mode) {
            setModeChipStyle(data.tracking_mode.current_mode);
            setText("cameraModeValue", data.tracking_mode.camera_mode);
            setText("trackingSourceValue", data.tracking_mode.tracking_source);
            setText("targetTypeValue", data.tracking_mode.target_type);
            setText("arucoStatusValue", data.tracking_mode.aruco_status);
            setText("modeReasonValue", data.tracking_mode.mode_reason);

            if (lastTrackingMode !== null && lastTrackingMode !== data.tracking_mode.current_mode) {
                addLog(`Tracking Mode changed: ${lastTrackingMode} → ${data.tracking_mode.current_mode}`);
            }

            lastTrackingMode = data.tracking_mode.current_mode;
        }

        if (data.recovery) {
            setText("mismatchCountValue", data.recovery.mismatch_count);
            setText("recoveryTriggerValue", data.recovery.recovery_trigger);
            setText("recoveryActionValue", data.recovery.recovery_action);
            setText("recoveryStateValue", data.recovery.recovery_state);
            setText("lastRecoveryValue", data.recovery.last_recovery);

            if (lastRecoveryState !== null && lastRecoveryState !== data.recovery.recovery_state) {
                addLog(`Recovery State changed: ${lastRecoveryState} → ${data.recovery.recovery_state}`);
            }

            lastRecoveryState = data.recovery.recovery_state;
        }

        if (data.control_ownership) {
            setText("controlOwnerValue", data.control_ownership.control_owner);
            setText("trackingEngineValue", data.control_ownership.tracking_engine);
            setText("conflictStatusValue", data.control_ownership.conflict_status);
            setText("controlPolicyValue", data.control_ownership.control_policy);

            if (lastConflictStatus !== null && lastConflictStatus !== data.control_ownership.conflict_status) {
                addLog(`Control Conflict Status changed: ${lastConflictStatus} → ${data.control_ownership.conflict_status}`);
            }

            lastConflictStatus = data.control_ownership.conflict_status;
        }

        if (data.debug) {
            setText("debugTargetValue", data.target_id);
            setText("totalDetectionsValue", data.debug.total_detections);
            setText("insideZoneValue", data.debug.inside_zone);
            setText("outsideZoneValue", data.debug.outside_zone);
            setText("trackStabilityValue", data.debug.track_stability);
            setText("lastReidValue", data.debug.last_reid);
            setText("idSwitchValue", data.debug.id_switch_count);
        }

        if (data.reid) {
            setText("reidStateValue", data.reid.state);
            setText("reidScoreValue", data.reid.score);
            setText("reidThresholdValue", data.reid.threshold);
            setText("reidEventValue", data.reid.event);
            setText("reidMethodValue", data.reid.method);
            setText("registeredTargetValue", data.reid.registered_target);
            setText("recoveryModeValue", data.reid.recovery_mode);

            if (lastReidEvent !== null && lastReidEvent !== data.reid.event) {
                addReidTimelineEvent(data.reid.state, data.reid.event, data.reid.score, data.reid.event_level);

                if (data.reid.event === "WRONG_TARGET_SUSPECTED") {
                    addLog(`Re-ID WARNING: ${data.reid.event}, score=${data.reid.score}`);
                } else {
                    addLog(`Re-ID Event: ${data.reid.event}, score=${data.reid.score}`);
                }
            }

            if (lastReidState !== null && lastReidState !== data.reid.state) {
                addLog(`Re-ID State changed: ${lastReidState} → ${data.reid.state}`);
            }

            lastReidEvent = data.reid.event;
            lastReidState = data.reid.state;
        }

        if (data.ptz_simulator) {
            setText("panDirectionValue", data.ptz_simulator.pan_direction);
            setText("tiltDirectionValue", data.ptz_simulator.tilt_direction);
            setText("zoomStateValue", data.ptz_simulator.zoom_state);
            setText("offsetXValue", data.ptz_simulator.offset_x);
            setText("offsetYValue", data.ptz_simulator.offset_y);
        }

        if (lastTargetId !== null && lastTargetId !== data.target_id) {
            addLog(`Target ID changed: ${lastTargetId} → ${data.target_id}`);
        }

        lastTargetId = data.target_id;

    } catch (error) {
        console.error("상태 조회 실패:", error);
    }
}

function drawBoxes(people) {
    const canvas = document.getElementById("overlay");
    const ctx = canvas.getContext("2d");

    canvas.width = canvas.offsetWidth;
    canvas.height = canvas.offsetHeight;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const centerX = canvas.width / 2;
    const centerY = canvas.height / 2;

    ctx.strokeStyle = "rgba(255, 255, 255, 0.75)";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(centerX - 25, centerY);
    ctx.lineTo(centerX + 25, centerY);
    ctx.moveTo(centerX, centerY - 25);
    ctx.lineTo(centerX, centerY + 25);
    ctx.stroke();

    ctx.fillStyle = "white";
    ctx.font = "14px Arial";
    ctx.fillText("CAMERA CENTER", centerX + 10, centerY - 10);

    ctx.strokeStyle = "rgba(0, 180, 255, 0.9)";
    ctx.lineWidth = 2;
    ctx.setLineDash([8, 6]);
    ctx.strokeRect(60, 50, 420, 220);
    ctx.setLineDash([]);

    ctx.fillStyle = "rgba(0, 180, 255, 0.9)";
    ctx.font = "14px Arial";
    ctx.fillText("ZONE LOCK AREA", 70, 45);

    people.forEach((p) => {
        const scaleX = canvas.width / 640;
        const scaleY = canvas.height / 480;

        const x = p.x * scaleX;
        const y = p.y * scaleY;
        const w = p.w * scaleX;
        const h = p.h * scaleY;

        let boxColor = p.inside ? "lime" : "red";

        if (p.reid_state === "SUSPENDED") boxColor = "orange";
        if (p.reid_state === "RECOVERED") boxColor = "cyan";

        ctx.strokeStyle = boxColor;
        ctx.lineWidth = p.target ? 5 : 3;
        ctx.strokeRect(x, y, w, h);

        ctx.fillStyle = boxColor;
        ctx.font = "14px Arial";
        ctx.fillText(`ID: ${p.id}`, x, y - 36);
        ctx.fillText(`Conf: ${p.confidence}`, x, y - 22);

        if (p.reid_score !== undefined) {
            ctx.fillText(`Re-ID: ${p.reid_score}`, x, y - 8);
        }

        if (p.target) {
            const targetCenterX = x + w / 2;
            const targetCenterY = y + h / 2;

            if (p.reid_state === "SUSPENDED") {
                ctx.fillStyle = "orange";
                ctx.fillText("SUSPENDED", x, y + h + 18);
            } else if (p.reid_state === "RECOVERED") {
                ctx.fillStyle = "cyan";
                ctx.fillText("RECOVERED", x, y + h + 18);
            } else {
                ctx.fillStyle = "yellow";
                ctx.fillText("TARGET", x, y + h + 18);
            }

            ctx.strokeStyle = ctx.fillStyle;
            ctx.lineWidth = 3;
            ctx.beginPath();
            ctx.moveTo(centerX, centerY);
            ctx.lineTo(targetCenterX, targetCenterY);
            ctx.stroke();

            const dx = targetCenterX - centerX;
            const dy = targetCenterY - centerY;

            let directionText = "HOLD";

            if (Math.abs(dx) > 35 || Math.abs(dy) > 35) {
                const horizontal = dx > 35 ? "RIGHT" : dx < -35 ? "LEFT" : "";
                const vertical = dy > 35 ? "DOWN" : dy < -35 ? "UP" : "";
                directionText = `${vertical} ${horizontal}`.trim();
            }

            ctx.fillStyle = p.reid_state === "SUSPENDED" ? "orange" : p.reid_state === "RECOVERED" ? "cyan" : "yellow";
            ctx.font = "16px Arial";
            ctx.fillText(`PTZ MOVE: ${directionText}`, 20, canvas.height - 20);
        }
    });
}

async function fetchDetections() {
    try {
        const response = await fetch("/api/detections");
        const data = await response.json();
        drawBoxes(data);
    } catch (error) {
        console.error("Detection fetch 실패:", error);
    }
}

document.querySelectorAll(".preset-grid .preset-btn").forEach((button) => {
    button.addEventListener("click", () => {
        document.querySelectorAll(".preset-grid .preset-btn").forEach((btn) => btn.classList.remove("active"));
        button.classList.add("active");
        addLog(`프리셋 선택: ${button.textContent}`);
    });
});

document.querySelectorAll(".position-btn").forEach((button) => {
    button.addEventListener("click", () => {
        document.querySelectorAll(".position-btn").forEach((btn) => btn.classList.remove("active"));
        button.classList.add("active");
        addLog(`위치 프리셋 선택: ${button.textContent}`);
    });
});

document.getElementById("startBtn").onclick = () => callAPI("/api/session/start", "Session Start");
document.getElementById("endBtn").onclick = () => callAPI("/api/session/end", "Session End");

document.getElementById("lockBtn").onclick = () => addLog("Target Lock 요청");
document.getElementById("unlockBtn").onclick = () => addLog("Target Unlock 요청");
document.getElementById("upBtn").onclick = () => addLog("PTZ Move Up");
document.getElementById("downBtn").onclick = () => addLog("PTZ Move Down");
document.getElementById("leftBtn").onclick = () => addLog("PTZ Move Left");
document.getElementById("rightBtn").onclick = () => addLog("PTZ Move Right");
document.getElementById("centerBtn").onclick = () => addLog("PTZ Home / Center");
document.getElementById("zoomInBtn").onclick = () => addLog("Zoom In");
document.getElementById("zoomOutBtn").onclick = () => addLog("Zoom Out");

document.getElementById("zoneBtn").onclick = () => callAPI("/api/zone/toggle", "Zone Lock Toggle");

fetchStatus();
fetchDetections();
addLog("Tracking Mode Dashboard initialized");

setInterval(fetchStatus, 1000);
setInterval(fetchDetections, 1000);
