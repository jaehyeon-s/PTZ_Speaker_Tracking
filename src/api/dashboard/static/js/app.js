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
    } catch (error) {
        addLog(`${message} 실패`);
        console.error(error);
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

document.getElementById("lockBtn").onclick = () => {
    callAPI("/api/ptz/lock", "Target Lock");
};

document.getElementById("unlockBtn").onclick = () => {
    callAPI("/api/ptz/unlock", "Target Unlock");
};

document.getElementById("upBtn").onclick = () => addLog("PTZ Move Up");
document.getElementById("downBtn").onclick = () => addLog("PTZ Move Down");
document.getElementById("leftBtn").onclick = () => addLog("PTZ Move Left");
document.getElementById("rightBtn").onclick = () => addLog("PTZ Move Right");
document.getElementById("centerBtn").onclick = () => addLog("PTZ Home / Center");

document.getElementById("zoomInBtn").onclick = () => addLog("Zoom In");
document.getElementById("zoomOutBtn").onclick = () => addLog("Zoom Out");
document.getElementById("zoneBtn").onclick = () => addLog("Zone Lock Toggle");

setTimeout(() => addLog("Dashboard UI initialized"), 800);
