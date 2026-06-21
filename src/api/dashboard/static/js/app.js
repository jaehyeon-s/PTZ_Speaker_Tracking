console.log("Dashboard Loaded");

let lastTargetId = null;
let lastReidEvent = null;
let lastReidState = null;
let lastHealthLevel = null;
let bboxEnabled = true;

function setText(id, value) {
    const element = document.getElementById(id);
    if (element) element.textContent = value;
}

function addLog(message) {
    console.log(message);
}

function setHealthLevelStyle(level) {
    const badge = document.getElementById("healthLevelValue");
    if (!badge) return;
    badge.textContent = level;
    badge.classList.remove("health-good", "health-caution", "health-warning");
    if (level === "GOOD") badge.classList.add("health-good");
    else if (level === "CAUTION") badge.classList.add("health-caution");
    else badge.classList.add("health-warning");
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
        setText("lastSystemActionValue", data.last_system_action || "NONE");

        bboxEnabled = Boolean(data.bbox_enabled);
        setText("bboxValue", bboxEnabled ? "ON" : "OFF");
        const bboxToggle = document.getElementById("bboxToggle");
        if (bboxToggle) bboxToggle.checked = bboxEnabled;

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

        if (data.reid) {
            setText("reidStateValue", data.reid.state);
            setText("reidScoreValue", data.reid.score);
            setText("reidThresholdValue", data.reid.threshold);
            setText("reidEventValue", data.reid.event);
            setText("reidMethodValue", data.reid.method);
            setText("registeredTargetValue", data.reid.registered_target);
            setText("recoveryModeValue", data.reid.recovery_mode);
            lastReidEvent = data.reid.event;
            lastReidState = data.reid.state;
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
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    canvas.width = canvas.offsetWidth;
    canvas.height = canvas.offsetHeight;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (!bboxEnabled) {
        ctx.fillStyle = "rgba(255,255,255,0.7)";
        ctx.font = "14px Arial";
        ctx.fillText("B-Box Overlay OFF", 20, 28);
        return;
    }

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
        if (p.reid_score !== undefined) ctx.fillText(`Re-ID: ${p.reid_score}`, x, y - 8);
        if (p.target) ctx.fillText(p.reid_state === "SUSPENDED" ? "SUSPENDED" : p.reid_state === "RECOVERED" ? "RECOVERED" : "TARGET", x, y + h + 18);
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

document.querySelectorAll(".position-btn").forEach((button) => {
    button.addEventListener("click", () => {
        document.querySelectorAll(".position-btn").forEach((btn) => btn.classList.remove("active"));
        button.classList.add("active");
        addLog(`위치 프리셋 선택: ${button.textContent}`);
    });
});

const bboxToggle = document.getElementById("bboxToggle");
if (bboxToggle) bboxToggle.onchange = () => callAPI("/api/view/bbox/toggle", "B-Box Toggle");

const startBtn = document.getElementById("startBtn");
if (startBtn) startBtn.onclick = () => callAPI("/api/session/start", "Session Start");
const endBtn = document.getElementById("endBtn");
if (endBtn) endBtn.onclick = () => callAPI("/api/session/end", "Session End");
const lockBtn = document.getElementById("lockBtn");
if (lockBtn) lockBtn.onclick = () => callAPI("/api/ptz/lock", "Target Lock");
const unlockBtn = document.getElementById("unlockBtn");
if (unlockBtn) unlockBtn.onclick = () => callAPI("/api/ptz/unlock", "Target Unlock");
const upBtn = document.getElementById("upBtn");
if (upBtn) upBtn.onclick = () => callAPI("/api/ptz/up", "PTZ Move Up");
const downBtn = document.getElementById("downBtn");
if (downBtn) downBtn.onclick = () => callAPI("/api/ptz/down", "PTZ Move Down");
const leftBtn = document.getElementById("leftBtn");
if (leftBtn) leftBtn.onclick = () => callAPI("/api/ptz/left", "PTZ Move Left");
const rightBtn = document.getElementById("rightBtn");
if (rightBtn) rightBtn.onclick = () => callAPI("/api/ptz/right", "PTZ Move Right");
const centerBtn = document.getElementById("centerBtn");
if (centerBtn) centerBtn.onclick = () => callAPI("/api/ptz/home", "PTZ Home / Center");
const zoomInBtn = document.getElementById("zoomInBtn");
if (zoomInBtn) zoomInBtn.onclick = () => callAPI("/api/ptz/zoom-in", "Zoom In");
const zoomOutBtn = document.getElementById("zoomOutBtn");
if (zoomOutBtn) zoomOutBtn.onclick = () => callAPI("/api/ptz/zoom-out", "Zoom Out");
const zoneBtn = document.getElementById("zoneBtn");
if (zoneBtn) zoneBtn.onclick = () => callAPI("/api/zone/toggle", "Zone Lock Toggle");
const resetBtn = document.getElementById("resetBtn");
if (resetBtn) resetBtn.onclick = () => {
    if (confirm("시스템 상태를 초기화할까요?")) callAPI("/api/system/reset", "System Reset");
};
const rebootBtn = document.getElementById("rebootBtn");
if (rebootBtn) rebootBtn.onclick = () => {
    if (confirm("재부팅 요청을 보낼까요? 실제 장비 연동 전에는 요청 상태만 기록됩니다.")) callAPI("/api/system/reboot", "Reboot Request");
};

fetchStatus();
fetchDetections();
setInterval(fetchStatus, 1000);
setInterval(fetchDetections, 1000);
