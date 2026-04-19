console.log("Dashboard Loaded");

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
    } catch (error) {
        console.error("상태 조회 실패:", error);
        addLog("상태 조회 실패");
    }
}

function drawBoxes(people) {
    const canvas = document.getElementById("overlay");
    const ctx = canvas.getContext("2d");

    canvas.width = canvas.offsetWidth;
    canvas.height = canvas.offsetHeight;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Zone Lock 영역 표시
    ctx.strokeStyle = "rgba(0, 180, 255, 0.9)";
    ctx.lineWidth = 2;
    ctx.setLineDash([8, 6]);
    ctx.strokeRect(60, 50, 420, 220);
    ctx.setLineDash([]);
    ctx.fillStyle = "rgba(0, 180, 255, 0.9)";
    ctx.font = "14px Arial";
    ctx.fillText("ZONE LOCK AREA", 70, 45);

    people.forEach((p) => {
        ctx.strokeStyle = p.inside ? "lime" : "red";
        ctx.lineWidth = p.target ? 5 : 3;
        ctx.strokeRect(p.x, p.y, p.w, p.h);

        ctx.fillStyle = p.inside ? "lime" : "red";
        ctx.font = "14px Arial";
        ctx.fillText(`ID: ${p.id}`, p.x, p.y - 8);

        if (p.target) {
            ctx.fillStyle = "yellow";
            ctx.fillText("TARGET", p.x, p.y + p.h + 18);
        }
    });
}

async function fetchDetections() {
    try {
        const response = await fetch("/api/detections");
        const data = await response.json();
        drawBoxes(data);

        const insideCount = data.filter((p) => p.inside).length;
        addDetectionSummary(data.length, insideCount);
    } catch (error) {
        console.error("Detection fetch 실패", error);
    }
}

let lastSummary = "";

function addDetectionSummary(total, inside) {
    const summary = `탐지 인원 ${total}명 / Zone 내부 ${inside}명`;
    if (summary !== lastSummary) {
        addLog(summary);
        lastSummary = summary;
    }
}

// 일반 프리셋 버튼 active
document.querySelectorAll(".preset-grid .preset-btn").forEach((button) => {
    button.addEventListener("click", () => {
        document.querySelectorAll(".preset-grid .preset-btn").forEach((btn) => btn.classList.remove("active"));
        button.classList.add("active");
        addLog(`프리셋 선택: ${button.textContent}`);
    });
});

// 위치 프리셋 버튼 active
document.querySelectorAll(".position-btn").forEach((button) => {
    button.addEventListener("click", () => {
        document.querySelectorAll(".position-btn").forEach((btn) => btn.classList.remove("active"));
        button.classList.add("active");
        addLog(`위치 프리셋 선택: ${button.textContent}`);
    });
});

// 세션 제어
document.getElementById("startBtn").onclick = () => {
    callAPI("/api/session/start", "Session Start");
};

document.getElementById("endBtn").onclick = () => {
    callAPI("/api/session/end", "Session End");
};

// PTZ 관련
document.getElementById("lockBtn").onclick = () => addLog("Target Lock 요청");
document.getElementById("unlockBtn").onclick = () => addLog("Target Unlock 요청");
document.getElementById("upBtn").onclick = () => addLog("PTZ Move Up");
document.getElementById("downBtn").onclick = () => addLog("PTZ Move Down");
document.getElementById("leftBtn").onclick = () => addLog("PTZ Move Left");
document.getElementById("rightBtn").onclick = () => addLog("PTZ Move Right");
document.getElementById("centerBtn").onclick = () => addLog("PTZ Home / Center");
document.getElementById("zoomInBtn").onclick = () => addLog("Zoom In");
document.getElementById("zoomOutBtn").onclick = () => addLog("Zoom Out");

// Zone Lock 토글
document.getElementById("zoneBtn").onclick = () => {
    callAPI("/api/zone/toggle", "Zone Lock Toggle");
};

// 초기 실행
fetchStatus();
fetchDetections();
addLog("Dashboard UI initialized");

// 주기 갱신
setInterval(fetchStatus, 1000);
setInterval(fetchDetections, 1000);
