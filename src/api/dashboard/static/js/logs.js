async function fetchLogs() {
    const box = document.getElementById("serverLogBox");
    if (!box) return;
    const response = await fetch("/api/logs");
    const data = await response.json();
    box.innerHTML = "";
    const logs = data.logs || [];
    logs.slice().reverse().forEach((log) => {
        const p = document.createElement("p");
        p.className = `server-log-item log-${(log.level || "INFO").toLowerCase()}`;
        p.textContent = `[${log.time}] [${log.level}] ${log.message}`;
        box.appendChild(p);
    });
}
fetchLogs();
setInterval(fetchLogs, 1000);
