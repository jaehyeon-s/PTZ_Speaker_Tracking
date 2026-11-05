"""Run a series of tracker tests against the Unity simulator's test-control API.

Per run: check /status is idle -> /run/start -> tracker for --tracker-seconds
-> /run/stop -> save frames.csv, summary.json and, for scenario runs,
truth.json and actors.csv. Stops at the first lost connection and prints the
UTC time.

The API token is read from the environment variable UNITY_TEST_TOKEN.

Run: python tools/unity_series.py --control http://HOST:8082 --camera-host HOST \\
       --scenario NAME [--scenario NAME ...] \\
       --series c1 "OPTIONS" --series c1zf "OPTIONS" --repeats 3
     python tools/unity_series.py --control http://HOST:8082 --list-scenarios

--pipeline pc runs the tracker on this computer: detector on the CPU, or on
the GPU with --pc-detector gpu; ReID is the Pi's model as float ONNX unless
--pc-reid yolo26n. --pipeline pi runs the Pi 5 NPU pipeline (Hailo HEF
detector and ReID). Result names are PREFIX_SCENARIO_LABEL_N.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import shlex
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "test_video"


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="milliseconds")


class Api:
    def __init__(self, base, token):
        self.base, self.token = base.rstrip("/"), token

    def call(self, method, path, timeout=10):
        request = urllib.request.Request(self.base + path, method=method, headers={"X-Token": self.token},
                                         data=b"" if method == "POST" else None)
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read()

    def json(self, method, path):
        return json.loads(self.call(method, path).decode("utf-8"))


def scenario_names(reply):
    """Scenario names from GET /scenarios: a list of names or of objects, or a dict holding one."""
    if isinstance(reply, dict):
        for key in ("scenarios", "items", "names"):
            if key in reply:
                return scenario_names(reply[key])
        return set(reply)
    names = set()
    for item in reply or []:
        if isinstance(item, dict):
            value = next((item[key] for key in ("name", "id", "scenario") if key in item), None)
            if value is not None:
                names.add(str(value))
        else:
            names.add(str(item))
    return names


def scenario_durations(reply):
    """durationSec per scenario from GET /scenarios ("details"), if the simulator gives it."""
    details = reply.get("details") if isinstance(reply, dict) else None
    return {
        str(item["name"]): float(item["durationSec"])
        for item in details or []
        if isinstance(item, dict) and item.get("name") and item.get("durationSec")
    }


def run_id_of(reply):
    for key in ("id", "run_id", "runId", "run"):
        if isinstance(reply, dict) and reply.get(key) not in (None, ""):
            value = reply[key]
            return str(value.get("id") if isinstance(value, dict) else value)
    raise RuntimeError(f"No run id in /run/start reply: {reply}")


def is_idle(status):
    state = str(status.get("state", status.get("status", ""))).lower()
    return state == "idle" or status.get("idle") is True or status.get("running") is False


# Person detector, tracker and speaker ReID per pipeline. The rest of the
# options (profile, recording, camera) is the same for both.
PIPELINES = {
    # Pi 5 + Hailo-8: the NPU-verified configuration.
    "pi": ["--person-tracker", "decoupled", "--hailo-hef", "yolo26n_h8.hef",
           "--tracker-config", "botsort-pi.yaml",
           "--speaker-identity-backend", "yolo-reid",
           "--speaker-reid-hef", "repvgg_a0_person_reid_512_h8.hef",
           "--speaker-reid-similarity-scale", "1.31", "--speaker-reid-similarity-offset", "-0.537"],
    # This computer: same decoupled detector/tracker stages; ReID per --pc-reid.
    "pc": ["--person-tracker", "decoupled", "--tracker-config", "botsort-pi.yaml",
           "--speaker-identity-backend", "yolo-reid"],
}
# --pc-reid: the Pi's ReID model as float ONNX with the Pi's calibration, or
# the earlier yolo26n-reid model.
PC_REID = {
    "repvgg": ["--speaker-reid-onnx", "repvgg_a0_person_reid_512.onnx",
               "--speaker-reid-similarity-scale", "1.31", "--speaker-reid-similarity-offset", "-0.537"],
    "yolo26n": ["--speaker-reid-model", "yolo26n-reid.onnx", "--speaker-reid-device", "cpu"],
}
# --pc-detector: the profile's NCNN model at 416 on the CPU, or the PyTorch
# model at 640 (the HEF's input size) on the GPU.
PC_DETECTORS = {
    "cpu": [],
    "gpu": ["--model", "yolo26n.pt", "--imgsz", "640", "--inference-device", "cuda"],
}


def pipeline_options(args):
    options = list(PIPELINES[args.pipeline])
    if args.pipeline == "pc":
        options += PC_REID[args.pc_reid] + PC_DETECTORS[args.pc_detector]
        if args.pc_reid == "repvgg" and args.pc_detector == "gpu":
            # 31 ms per crop on this CPU against about 5 ms on the Hailo NPU.
            options += ["--speaker-reid-device", "cuda"]
    if getattr(args, "max_process_fps", 0):
        options += ["--max-process-fps", str(args.max_process_fps)]
    return options


RETRY_SECONDS = 15


def wait_until_idle(api, name, args):
    """Wait while another user's run (or a busy simulator) holds the simulator."""
    deadline = time.monotonic() + args.wait_seconds
    while True:
        try:
            status = api.json("GET", "/status")
            if is_idle(status):
                return status
            reason = f"{status.get('state')} {status.get('label')}"
        except urllib.error.HTTPError as error:  # e.g. 500 while the simulator's main thread is busy
            reason = f"HTTP {error.code}"
        if time.monotonic() > deadline:
            raise RuntimeError(f"Simulator not idle for {args.wait_seconds:.0f} s before {name}: {reason}")
        print(f"waiting before {name}: {reason}", flush=True)
        time.sleep(RETRY_SECONDS)


def start_run(api, name, parameters, args):
    """POST /run/start; a 409 means another run holds the simulator, so wait and try again."""
    deadline = time.monotonic() + args.wait_seconds
    while True:
        status = wait_until_idle(api, name, args)
        try:
            return status, api.json("POST", f"/run/start?{urllib.parse.urlencode(parameters)}")
        except urllib.error.HTTPError as error:
            if error.code != 409 or time.monotonic() > deadline:
                raise
            print(f"start of {name} refused (409, another run); retrying", flush=True)
            time.sleep(RETRY_SECONDS)


def start_conditions(status):
    """Camera set-up at the start of a run, to find runs made under other conditions later."""
    ptz = status.get("ptz") or {}
    return {"preset": status.get("preset"), "hfov": ptz.get("hfov"), "pan": ptz.get("pan"),
            "tilt": ptz.get("tilt"), "zoom_pos": ptz.get("zoom_pos"),
            "ptz_position": status.get("ptz_position"), "marker_mode": status.get("marker_mode"),
            "stream_fps": status.get("stream_fps")}


def check_hfov(status, required, name):
    hfov = (status.get("ptz") or {}).get("hfov")
    if required is not None and (hfov is None or abs(float(hfov) - required) > 0.5):
        raise RuntimeError(f"Simulator hfov is {hfov}, not {required:g}, before {name}")


def one_run(api, name, tracker_options, args, scenario="", label=""):
    parameters = {"label": name}
    if args.owner:
        parameters["owner"] = args.owner
    scenario_duration = args.scenario_durations.get(scenario) if scenario else None
    if args.duration is not None:
        parameters["duration"] = args.duration
    elif scenario_duration:
        parameters["duration"] = int(round(scenario_duration))
    if args.path:
        parameters["path"] = args.path
    if scenario:
        parameters["scenario"] = scenario
    if args.require_hfov is not None:
        # The simulator puts the camera back on its preset when a run starts,
        # so a zoom left over from the previous run is only reported here; the
        # check that stops the series is the one right after the start.
        try:
            check_hfov(wait_until_idle(api, name, args), args.require_hfov, name)
        except RuntimeError as error:
            print(f"note: {error}; checking again after the start", flush=True)
    status, started = start_run(api, name, parameters, args)
    run_id = run_id_of(started)
    running = api.json("GET", "/status")
    if args.require_hfov is not None:
        try:
            check_hfov(running, args.require_hfov, name)
        except RuntimeError:
            api.call("POST", "/run/stop")
            raise
    if args.path and running.get("path") != args.path:
        api.call("POST", "/run/stop")
        raise RuntimeError(f"Simulator runs path {running.get('path')!r}, not {args.path!r}")
    # Without --duration the simulator runs the scenario's own length; the
    # tracker stops 2 s before the end.
    duration = (
        args.duration
        or scenario_duration
        or (started.get("duration_sec") if isinstance(started, dict) else None)
        or running.get("duration_sec")
    )
    tracker_seconds = args.tracker_seconds
    if tracker_seconds is None:
        if not duration:
            api.call("POST", "/run/stop")
            raise RuntimeError(f"No duration_sec in the /run/start reply for {name}: {started}")
        tracker_seconds = float(duration) - 2.0
    meta = {"name": name, "run_id": run_id, "start_utc": utc(), "status_before": status,
            "duration_sec": duration, "tracker_seconds": tracker_seconds,
            "start_conditions": start_conditions(running),
            "status_after_start": running,
            "start_reply": started, "tracker_options": tracker_options,
            "scenario": scenario, "label": label, "pipeline": args.pipeline,
            "pipeline_options": " ".join(pipeline_options(args))}
    command = [sys.executable, "-m", "marker_tracker.app", "--runtime-profile", "pi-qon-http",
               *pipeline_options(args),
               "--record-input-video", str(OUT / "raw" / f"{name}.mp4"),
               "--record-input-fps", str(args.record_fps),
               "--identity-trace-file", str(OUT / "traces" / f"{name}.jsonl"),
               "--rtsp-url", f"http://{args.camera_host}:8081/video",
               "--qon-camera-url", f"http://{args.camera_host}:8080", "--qon-auth-mode", "none",
               "--qon-log-commands", "--no-display", *shlex.split(tracker_options)]
    with (OUT / "logs" / f"{name}.log").open("w") as log:
        tracker = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        meta["tracker_start_utc"] = utc()
        try:
            tracker.wait(timeout=tracker_seconds)
            meta["tracker_exit"] = tracker.returncode  # ended early: failed to start or lost the stream
        except subprocess.TimeoutExpired:
            tracker.send_signal(signal.SIGINT)
            try:
                tracker.wait(timeout=10)
            except subprocess.TimeoutExpired:
                tracker.kill()
            meta["tracker_exit"] = "stopped_after_timeout"
    meta["tracker_end_utc"] = utc()
    try:
        meta["status_during"] = api.json("GET", "/status")
        meta["stop_reply"] = api.json("POST", "/run/stop")
        directory = OUT / "unity_metrics"
        (directory / f"{name}_frames.csv").write_bytes(api.call("GET", f"/run/{run_id}/frames.csv", timeout=60))
        (directory / f"{name}_summary.json").write_bytes(api.call("GET", f"/run/{run_id}/summary.json"))
        if scenario:
            # Ground truth of a scenario run: presenter timetable and per-person boxes.
            for record in ("truth.json", "actors.csv"):
                try:
                    data = api.call("GET", f"/run/{run_id}/{record}", timeout=60)
                except urllib.error.HTTPError as error:
                    meta.setdefault("missing_records", []).append(f"{record}: HTTP {error.code}")
                    continue
                (directory / f"{name}_{record}").write_bytes(data)
    finally:
        meta["end_utc"] = utc()
        (OUT / "unity_metrics" / f"{name}_meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    if meta["tracker_exit"] != "stopped_after_timeout":
        raise RuntimeError(f"Tracker ended early in {name} (exit {meta['tracker_exit']}); see test_video/logs/{name}.log")
    failed_sends = (OUT / "logs" / f"{name}.log").read_text(errors="replace").count("Failed to send")
    if failed_sends > 5:
        raise RuntimeError(f"{failed_sends} failed PTZ sends in {name}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--control", required=True, help="Test-control API base URL")
    parser.add_argument("--camera-host")
    parser.add_argument("--series", nargs=2, action="append", metavar=("LABEL", "TRACKER_OPTIONS"))
    parser.add_argument("--scenario", action="append", default=[],
                        help="Scenario passed to /run/start (repeat for several; see --list-scenarios)")
    parser.add_argument("--list-scenarios", action="store_true", help="Print GET /scenarios and exit")
    parser.add_argument("--pipeline", choices=tuple(PIPELINES), default="pc")
    parser.add_argument("--pc-detector", choices=tuple(PC_DETECTORS), default="cpu")
    parser.add_argument("--pc-reid", choices=tuple(PC_REID), default="repvgg")
    parser.add_argument("--record-fps", type=float, default=15.0, help="--record-input-fps of the tracker")
    parser.add_argument("--max-process-fps", type=float, default=0.0,
                        help="--max-process-fps of the tracker for every series, e.g. 21.9 to run a PC at the Pi's rate")
    parser.add_argument("--owner", default="codex", help="owner passed to /run/start (empty: none)")
    parser.add_argument("--require-hfov", type=float, default=None,
                        help="Stop unless /status reports this horizontal FOV before and at the start of each run")
    parser.add_argument("--wait-seconds", type=float, default=1800.0,
                        help="How long to wait for the simulator to become free before giving up")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--prefix", default="series")
    parser.add_argument("--duration", type=int, default=None,
                        help="Run length in seconds; default: the scenario's durationSec from GET /scenarios")
    parser.add_argument("--path", default="", help="Presenter path preset passed to /run/start")
    parser.add_argument("--tracker-seconds", type=float, default=None,
                        help="Default: the run length minus 2 s")
    args = parser.parse_args(argv)
    token = os.environ.get("UNITY_TEST_TOKEN")
    if not token:
        parser.error("Set UNITY_TEST_TOKEN")
    if args.list_scenarios:
        print(json.dumps(Api(args.control, token).json("GET", "/scenarios"), indent=1, ensure_ascii=False))
        return 0
    if not args.series or not args.camera_host:
        parser.error("--series and --camera-host are required")
    for directory in ("raw", "traces", "logs", "unity_metrics"):
        (OUT / directory).mkdir(parents=True, exist_ok=True)
    api = Api(args.control, token)
    args.scenario_durations = {}
    if args.scenario:
        known = api.json("GET", "/scenarios")
        args.scenario_durations = scenario_durations(known)
        names = scenario_names(known)
        unknown = [scenario for scenario in args.scenario if names and scenario not in names]
        if unknown:
            parser.error(f"Unknown scenario(s) {unknown}; the simulator offers {sorted(names)}")
    # Interleave scenarios and series so that slow drift in the simulator
    # affects all of them alike.
    runs = [
        (repeat, scenario, label, options)
        for repeat in range(1, args.repeats + 1)
        for scenario in (args.scenario or [""])
        for label, options in args.series
    ]
    for repeat, scenario, label, options in runs:
        name = "_".join(part for part in (args.prefix, scenario, label, str(repeat)) if part)
        if (OUT / "unity_metrics" / f"{name}_summary.json").exists():
            print(f"skip {name}: already recorded", flush=True)
            continue
        try:
            one_run(api, name, options, args, scenario=scenario, label=label)
        except (urllib.error.URLError, OSError, RuntimeError, ValueError) as error:
            print(f"STOPPED at {utc()} during {name}: {error}", flush=True)
            try:
                api.call("POST", "/run/stop")
            except (urllib.error.URLError, OSError):
                pass
            return 3
        print(f"done {name} at {utc()}", flush=True)
        time.sleep(2)
    print("ALL_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
