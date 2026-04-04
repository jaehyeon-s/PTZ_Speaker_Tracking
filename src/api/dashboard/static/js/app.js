console.log("Dashboard Loaded");

// 로그 추가 함수
function addLog(message) {
    const logBox = document.getElementById("log-box");
    const p = document.createElement("p");

    const time = new Date().toLocaleTimeString();
    p.textContent = `[${time}] ${message}`;

    logBox.appendChild(p);
}

// API 호출 함수
async function callAPI(url, message) {
    try {
        const response = await fetch(url, { method: "POST" });
        const data = await response.json();

        addLog(message + " 성공");
    } catch (error) {
        addLog(message + " 실패");
        console.error(error);
    }
}

// 버튼 이벤트 연결
document.getElementById("startBtn").onclick = () => {
    callAPI("/api/session/start", "Session Start");
};

document.getElementById("endBtn").onclick = () => {
    callAPI("/api/session/end", "Session End");
};

document.getElementById("lockBtn").onclick = () => {
    callAPI("/api/ptz/lock", "Target Lock");
};

document.getElementById("unlockBtn").onclick = () => {
    callAPI("/api/ptz/unlock", "Target Unlock");
};

// PTZ 방향 (더미 로그)
document.getElementById("upBtn").onclick = () => addLog("PTZ Move Up");
document.getElementById("downBtn").onclick = () => addLog("PTZ Move Down");
document.getElementById("leftBtn").onclick = () => addLog("PTZ Move Left");
document.getElementById("rightBtn").onclick = () => addLog("PTZ Move Right");

// Zoom
document.getElementById("zoomInBtn").onclick = () => addLog("Zoom In");
document.getElementById("zoomOutBtn").onclick = () => addLog("Zoom Out");

// Zone Lock
document.getElementById("zoneBtn").onclick = () => addLog("Zone Lock Toggle");
