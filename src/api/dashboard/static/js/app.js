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

        // API 호출 후 상태 다시 반영
        fetchStatus();
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
    } catch (error) {
        console.error("상태 조회 실패:", error);
        addLog("상태 조회 실패");
    }
}

// 일반 프리셋 버튼 active 전환
document.querySelectorAll(".preset-grid .preset-btn").forEach((button) => {
    button.addEventListener("click", () => {
        document.querySelectorAll(".preset-grid .preset-btn").forEach((btn) => btn.classList.remove("active"));
        button.classList.add("active");
        addLog(`프리셋 선택: ${button.textContent}`);
    });
});

// 위치 프리셋 버튼 active 전환
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

// PTZ 제어
document.getElementById("lockBtn").onclick = () => {
    addLog("Target Lock 요청");
};

document.getElementById("unlockBtn").onclick = () => {
    addLog("Target Unlock 요청");
};

// 방향키 제어
document.getElementById("upBtn").onclick = () => addLog("PTZ Move Up");
document.getElementById("downBtn").onclick = () => addLog("PTZ Move Down");
document.getElementById("leftBtn").onclick = () => addLog("PTZ Move Left");
document.getElementById("rightBtn").onclick = () => addLog("PTZ Move Right");
document.getElementById("centerBtn").onclick = () => addLog("PTZ Home / Center");

// 줌 / 존락
document.getElementById("zoomInBtn").onclick = () => addLog("Zoom In");
document.getElementById("zoomOutBtn").onclick = () => addLog("Zoom Out");
document.getElementById("zoneBtn").onclick = () => addLog("Zone Lock Toggle");

// 최초 1회 실행
fetchStatus();
addLog("Dashboard UI initialized");

// 1초마다 상태 갱신
setInterval(fetchStatus, 1000);
