console.log("Dashboard Loaded");

// 로그 추가 테스트
function addLog(message) {
    const logBox = document.getElementById("log-box");
    const p = document.createElement("p");
    p.textContent = "[INFO] " + message;
    logBox.appendChild(p);
}

// 테스트용
setTimeout(() => addLog("System initialized"), 2000);
