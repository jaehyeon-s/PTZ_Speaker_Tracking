console.log("Dashboard Loaded");

// 로그 추가 함수
function addLog(message) {
    const logBox = document.getElementById("log-box");
    const p = document.createElement("p");
    const time = new Date().toLocaleTimeString();
    p.textContent = `[${time}] ${message}`;
    logBox.appendChild(p);
    logBox.scrollTop = logBox.scrollHeight;
}

// API 호출 함수
async function callAPI(url, message) {
    try {
        const response = await fetch(url, { method: "POST" });
        const data = await response.json();
        addLog(`${message} 성공`);
        console.log(data);
    } catch (error) {
        addLog(`${message} 실패`);
        console.error(error);
    }
}

// 모드 버튼 active 전환
document.querySelectorAll(".mode-btn").forEach((button) => {
    button.addEventListener("click", () => {
        document.querySelectorAll(".mode-btn").forEach((btn) => btn.classList.remove("active"));
        button.classList.add("active");
        addLog(`카메라 모드 변경: ${button.textContent}`);
    });
});

// 서브 탭 active 전환
document.querySelectorAll(".sub-tab").forEach((button) => {
    button.addEventListener("click", () => {
        document.querySelectorAll(".sub-tab").forEach((btn) => btn.classList.remove("active"));
        button.classList.add("active");
        addLog(`탭 변경: ${button.textContent}`);
    });
});

// 프리셋 버튼 active 전환
document.querySelectorAll(".preset-btn").forEach((button) => {
    button.addEventListener("click", () => {
        document.querySelectorAll(".preset-btn").forEach((btn) => btn.classList.remove("active"));
        button.classList.add("active");
        addLog(`프리셋 선택: ${button.textContent}`);
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
    callAPI("/api/ptz/lock", "Target Lock");
};

document.getElementById("unlockBtn").onclick = () => {
    callAPI("/api/ptz/unlock", "Target Unlock");
};

// 방향키는 우선 더미 로그
document.getElementById("upBtn").onclick = () => addLog("PTZ Move Up");
document.getElementById("downBtn").onclick = () => addLog("PTZ Move Down");
document.getElementById("leftBtn").onclick = () => addLog("PTZ Move Left");
document.getElementById("rightBtn").onclick = () => addLog("PTZ Move Right");
document.getElementById("centerBtn").onclick = () => addLog("PTZ Center / Home");

// 줌/존락
document.getElementById("zoomInBtn").onclick = () => addLog("Zoom In");
document.getElementById("zoomOutBtn").onclick = () => addLog("Zoom Out");
document.getElementById("zoneBtn").onclick = () => addLog("Zone Lock Toggle");

// 시작 로그
setTimeout(() => addLog("Dashboard UI initialized"), 800);
