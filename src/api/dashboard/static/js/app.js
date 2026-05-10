console.log("Dashboard Loaded");

let lastTargetId = null;

function addLog(message) {
    const logBox = document.getElementById("log-box");
    const p = document.createElement("p");
    const time = new Date().toLocaleTimeString();
    p.textContent = `[${time}] ${message}`;
    logBox.appendChild(p);
    logBox.scrollTop = logBox.scrollHeight;
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

        document.getElementById("detectorValue").textContent = data.detector;
        document.getElementById("trackerValue").textContent = data.tracker;
        document.getElementById("fpsValue").textContent = data.fps;
        document.getElementById("targetIdValue").textContent = data.target_id;
        document.getElementById("zoneLockValue").textContent = data.zone_lock;
        document.getElementById("ptzValue").textContent = data.ptz_status;
        document.getElementById("sessionValue").textContent = data.session_state;
        document.getElementById("sessionBadge").textContent = `Session ${data.session_state}`;

        if (data.debug) {
            document.getElementById("debugTargetValue").textContent = data.target_id;
            document.getElementById("totalDetectionsValue").textContent = data.debug.total_detections;
            document.getElementById("insideZoneValue").textContent = data.debug.inside_zone;
            document.getElementById("outsideZoneValue").textContent = data.debug.outside_zone;
            document.getElementById("trackStabilityValue").textContent = data.debug.track_stability;
            document.getElementById("lastReidValue").textContent = data.debug.last_reid;
            document.getElementById("idSwitchValue").textContent = data.debug.id_switch_count;
        }

        if (data.ptz_simulator) {
            document.getElementById("panDirectionValue").textContent = data.ptz_simulator.pan_direction;
            document.getElementById("tiltDirectionValue").textContent = data.ptz_simulator.tilt_direction;
            document.getElementById("zoomStateValue").textContent = data.ptz_simulator.zoom_state;
            document.getElementById("offsetXValue").textContent = data.ptz_simulator.offset_x;
            document.getElementById("offsetYValue").textContent = data.ptz_simulator.offset_y;
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

    // 카메라 중심 십자선
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

    // Zone Lock 영역
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

        ctx.strokeStyle = p.inside ? "lime" : "red";
        ctx.lineWidth = p.target ? 5 : 3;
        ctx.strokeRect(x, y, w, h);

        ctx.fillStyle = p.inside ? "lime" : "red";
        ctx.font = "14px Arial";
        ctx.fillText(`ID: ${p.id}`, x, y - 22);
        ctx.fillText(`Conf: ${p.confidence}`, x, y - 8);

        if (p.target) {
            const targetCenterX = x + w / 2;
            const targetCenterY = y + h / 2;

            ctx.fillStyle = "yellow";
            ctx.fillText("TARGET", x, y + h + 18);

            // 중심에서 타겟까지 방향선
            ctx.strokeStyle = "yellow";
            ctx.lineWidth = 3;
            ctx.beginPath();
            ctx.moveTo(centerX, centerY);
            ctx.lineTo(targetCenterX, targetCenterY);
            ctx.stroke();

            // 방향 텍스트
            const dx = targetCenterX - centerX;
            const dy = targetCenterY - centerY;

            let directionText = "HOLD";

            if (Math.abs(dx) > 35 || Math.abs(dy) > 35) {
                const horizontal = dx > 35 ? "RIGHT" : dx < -35 ? "LEFT" : "";
                const vertical = dy > 35 ? "DOWN" : dy < -35 ? "UP" : "";
                directionText = `${vertical} ${horizontal}`.trim();
            }

            ctx.fillStyle = "yellow";
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
        console.error("Detection fetch 실패", error);
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

document.getElementById("startBtn").onclick = () => {
    callAPI("/api/session/start", "Session Start");
};

document.getElementById("endBtn").onclick = () => {
    callAPI("/api/session/end", "Session End");
};

document.getElementById("lockBtn").onclick = () => addLog("Target Lock 요청");
document.getElementById("unlockBtn").onclick = () => addLog("Target Unlock 요청");

document.getElementById("upBtn").onclick = () => addLog("PTZ Move Up");
document.getElementById("downBtn").onclick = () => addLog("PTZ Move Down");
document.getElementById("leftBtn").onclick = () => addLog("PTZ Move Left");
document.getElementById("rightBtn").onclick = () => addLog("PTZ Move Right");
document.getElementById("centerBtn").onclick = () => addLog("PTZ Home / Center");

document.getElementById("zoomInBtn").onclick = () => addLog("Zoom In");
document.getElementById("zoomOutBtn").onclick = () => addLog("Zoom Out");

document.getElementById("zoneBtn").onclick = () => {
    callAPI("/api/zone/toggle", "Zone Lock Toggle");
};

fetchStatus();
fetchDetections();
addLog("Virtual PTZ Simulator UI initialized");

setInterval(fetchStatus, 1000);
setInterval(fetchDetections, 1000);
