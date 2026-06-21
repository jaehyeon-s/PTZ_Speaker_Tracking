from fastapi import APIRouter
from src.api.state import app_state
from src.api.services.mvpv3_adapter import load_mvpv3_dashboard_state
from src.api.services.command_bus import command_bridge_status, write_command
import time
import math

router = APIRouter()


def ensure_base_state():
    app_state.setdefault("detector", "YOLO26n (NCNN)")
    app_state.setdefault("tracker", "ByteTrack")
    app_state.setdefault("fps", 27.2)
    app_state.setdefault("target_id", "None")
    app_state.setdefault("zone_lock", "ON")
    app_state.setdefault("ptz_status", "READY")
    app_state.setdefault("session_state", "IDLE")


def get_scenario():
    t = time.time()
    phase = int(t) % 40

    if phase < 10:
        return {
            "tracking_mode": "ZONE_TRACKING",
            "camera_mode": "PRESENTER",
            "tracking_source": "Re-ID + Zone Lock",
            "target_type": "Person",
            "aruco_status": "NOT_USED",
            "mode_reason": "Professor detected inside lecture zone",
            "reid_state": "ACTIVE",
            "reid_score": 0.81,
            "reid_event": "TARGET_MATCHED",
            "reid_event_level": "INFO",
            "recovery_mode": "OFF",
            "mismatch_count": 0,
            "recovery_trigger": "OFF",
            "recovery_action": "NONE",
            "recovery_state": "IDLE",
            "last_recovery": "-",
            "control_owner": "Pi Supervisor",
            "tracking_engine": "Qon4K6012XN Auto Tracking",
            "conflict_status": "NONE",
            "control_policy": "Pi monitors mismatch, camera performs tracking",
            "demo_step": 1,
            "demo_action": "Presenter registered",
        }

    elif phase < 20:
        return {
            "tracking_mode": "CLASS_TRACKING",
            "camera_mode": "PRESENTER",
            "tracking_source": "YOLO Class",
            "target_type": "Person/Class",
            "aruco_status": "NOT_USED",
            "mode_reason": "Class-based target tracking test",
            "reid_state": "ACTIVE",
            "reid_score": 0.74,
            "reid_event": "TARGET_SCORE_FLUCTUATION",
            "reid_event_level": "INFO",
            "recovery_mode": "OFF",
            "mismatch_count": 1,
            "recovery_trigger": "OFF",
            "recovery_action": "MONITORING",
            "recovery_state": "WATCHING",
            "last_recovery": "-",
            "control_owner": "Pi Supervisor",
            "tracking_engine": "YOLO Class + Camera Tracking",
            "conflict_status": "NONE",
            "control_policy": "Class result is used as tracking reference",
            "demo_step": 2,
            "demo_action": "Auto Tracking monitoring",
        }

    elif phase < 30:
        return {
            "tracking_mode": "MARKER_TRACKING",
            "camera_mode": "ZONE",
            "tracking_source": "ArUco Marker",
            "target_type": "Marker",
            "aruco_status": "NOT_DETECTED",
            "mode_reason": "Marker recognition failed, fallback required",
            "reid_state": "SUSPENDED",
            "reid_score": 0.59,
            "reid_event": "WRONG_TARGET_SUSPECTED",
            "reid_event_level": "WARNING",
            "recovery_mode": "ON",
            "mismatch_count": 3,
            "recovery_trigger": "ON",
            "recovery_action": "SWITCH_TO_ZONE_MODE",
            "recovery_state": "RECOVERING",
            "last_recovery": "Zone mode fallback",
            "control_owner": "Pi Supervisor",
            "tracking_engine": "Zone Lock Recovery",
            "conflict_status": "PREVENTED",
            "control_policy": "Pi blocks wrong target and switches camera to zone mode",
            "demo_step": 3,
            "demo_action": "Mismatch detected, zone fallback",
        }

    else:
        return {
            "tracking_mode": "RECOVERY_TRACKING",
            "camera_mode": "ZONE",
            "tracking_source": "Re-ID Recovery + Zone Lock",
            "target_type": "Person",
            "aruco_status": "NOT_USED",
            "mode_reason": "Target recovered after mismatch detection",
            "reid_state": "RECOVERED",
            "reid_score": 0.78,
            "reid_event": "TARGET_RECOVERED",
            "reid_event_level": "SUCCESS",
            "recovery_mode": "OFF",
            "mismatch_count": 0,
            "recovery_trigger": "OFF",
            "recovery_action": "RECOVERY_COMPLETED",
            "recovery_state": "COMPLETED",
            "last_recovery": "Target restored",
            "control_owner": "Pi Supervisor",
            "tracking_engine": "Qon4K6012XN Auto Tracking",
            "conflict_status": "NONE",
            "control_policy": "Camera tracking resumed after recovery",
            "demo_step": 4,
            "demo_action": "Recovery completed",
        }


def calc_health_score(scenario, track_stability, pan_direction, tilt_direction):
    score = 100

    if scenario["reid_state"] == "SUSPENDED":
        score -= 30
    elif scenario["reid_state"] == "RECOVERED":
        score -= 8

    if scenario["reid_score"] < 0.70:
        score -= 20
    elif scenario["reid_score"] < 0.78:
        score -= 8

    if track_stability == "WARNING":
        score -= 15

    if scenario["conflict_status"] != "NONE":
        score -= 10

    if scenario["aruco_status"] == "NOT_DETECTED":
        score -= 8

    if pan_direction != "HOLD" or tilt_direction != "HOLD":
        score -= 5

    return max(score, 0)


def calc_integration_readiness(scenario):
    score = 100

    if scenario["aruco_status"] == "NOT_DETECTED":
        score -= 10

    if scenario["reid_state"] == "SUSPENDED":
        score -= 15

    if scenario["conflict_status"] == "PREVENTED":
        score -= 5

    osnet_backend = "STANDBY"
    osnet_note = "ONNX model path and threshold are not applied yet"

    return {
        "score": max(score, 0),
        "level": "READY" if score >= 85 else "CHECK_REQUIRED",
        "data_source": "MOCK_FALLBACK",
        "backend_connection": "STANDBY",
        "vision_module": "READY",
        "reid_backend": "HSV_BASELINE",
        "osnet_backend": osnet_backend,
        "osnet_note": osnet_note,
        "camera_module": "Qon4K6012XN_READY",
        "ptz_loop": "SIMULATION_READY",
        "api_contract": "READY",
        "next_action": "Replace mock status payload with team module outputs",
    }


def build_mock_detections():
    ensure_base_state()

    t = time.time()
    offset = int(70 * math.sin(t))
    scenario = get_scenario()

    detections = [
        {
            "id": 1,
            "x": 210 + offset,
            "y": 95,
            "w": 90,
            "h": 170,
            "inside": True,
            "target": True,
            "confidence": 0.92,
            "reid_score": scenario["reid_score"],
            "reid_state": scenario["reid_state"],
        },
        {
            "id": 2,
            "x": 420 - offset,
            "y": 120,
            "w": 95,
            "h": 160,
            "inside": False,
            "target": False,
            "confidence": 0.81,
            "reid_score": 0.69,
            "reid_state": "CANDIDATE",
        },
    ]

    target = next((d for d in detections if d["target"]), None)

    frame_w = 640
    frame_h = 480
    center_x = frame_w // 2
    center_y = frame_h // 2

    if target:
        target_center_x = target["x"] + target["w"] // 2
        target_center_y = target["y"] + target["h"] // 2

        offset_x = target_center_x - center_x
        offset_y = target_center_y - center_y

        pan_direction = "HOLD"
        tilt_direction = "HOLD"

        if offset_x > 40:
            pan_direction = "RIGHT"
        elif offset_x < -40:
            pan_direction = "LEFT"

        if offset_y > 40:
            tilt_direction = "DOWN"
        elif offset_y < -40:
            tilt_direction = "UP"

        app_state["target_id"] = str(target["id"])
    else:
        offset_x = 0
        offset_y = 0
        pan_direction = "HOLD"
        tilt_direction = "HOLD"
        app_state["target_id"] = "None"

    inside_count = len([d for d in detections if d["inside"]])
    track_stability = "GOOD" if scenario["reid_state"] in ["ACTIVE", "RECOVERED"] else "WARNING"

    health_score = calc_health_score(
        scenario,
        track_stability,
        pan_direction,
        tilt_direction
    )

    if health_score >= 85:
        health_level = "GOOD"
    elif health_score >= 65:
        health_level = "CAUTION"
    else:
        health_level = "WARNING"

    warning_count = 1 if scenario["reid_state"] == "SUSPENDED" else 0
    integration = calc_integration_readiness(scenario)

    app_state["debug"] = {
        "total_detections": len(detections),
        "inside_zone": inside_count,
        "outside_zone": len(detections) - inside_count,
        "track_stability": track_stability,
        "last_reid": "2.1 sec ago",
        "id_switch_count": 0 if scenario["reid_state"] != "SUSPENDED" else 1,
    }

    app_state["ptz_simulator"] = {
        "frame_center_x": center_x,
        "frame_center_y": center_y,
        "offset_x": offset_x,
        "offset_y": offset_y,
        "pan_direction": pan_direction,
        "tilt_direction": tilt_direction,
        "zoom_state": "HOLD",
    }

    app_state["reid"] = {
        "state": scenario["reid_state"],
        "score": scenario["reid_score"],
        "threshold": 0.70,
        "event": scenario["reid_event"],
        "event_level": scenario["reid_event_level"],
        "method": "HSV Histogram / OSNet-style ONNX Backend",
        "registered_target": "Professor",
        "recovery_mode": scenario["recovery_mode"],
    }

    app_state["health"] = {
        "score": health_score,
        "level": health_level,
        "tracking": track_stability,
        "reid": scenario["reid_state"],
        "ptz": "TRACKING" if pan_direction != "HOLD" or tilt_direction != "HOLD" else "HOLD",
        "warnings": warning_count,
    }

    app_state["tracking_mode"] = {
        "current_mode": scenario["tracking_mode"],
        "camera_mode": scenario["camera_mode"],
        "tracking_source": scenario["tracking_source"],
        "target_type": scenario["target_type"],
        "aruco_status": scenario["aruco_status"],
        "mode_reason": scenario["mode_reason"],
    }

    app_state["recovery"] = {
        "mismatch_count": scenario["mismatch_count"],
        "recovery_trigger": scenario["recovery_trigger"],
        "recovery_action": scenario["recovery_action"],
        "recovery_state": scenario["recovery_state"],
        "last_recovery": scenario["last_recovery"],
    }

    app_state["control_ownership"] = {
        "control_owner": scenario["control_owner"],
        "tracking_engine": scenario["tracking_engine"],
        "conflict_status": scenario["conflict_status"],
        "control_policy": scenario["control_policy"],
    }

    app_state["integration"] = integration

    app_state["module_connection"] = {
        "vision_backend": "READY",
        "bytetrack": "READY",
        "zone_lock": "READY",
        "reid_hsv": "READY",
        "reid_osnet": "STANDBY",
        "aruco_marker": scenario["aruco_status"],
        "ptz_camera": "READY",
        "dashboard_api": "READY",
    }

    app_state["api_contract"] = {
        "status_endpoint": "/api/status OK",
        "detections_endpoint": "/api/detections OK",
        "required_fields": "detections, tracking_mode, recovery, control_ownership, integration",
        "missing_fields": "NONE",
        "integration_note": "Team modules can replace mock payload without changing dashboard UI",
    }

    app_state["demo_flow"] = {
        "current_step": scenario["demo_step"],
        "current_action": scenario["demo_action"],
        "presenter_registration": "READY",
        "camera_presenter_mode": "READY",
        "auto_tracking": "ON",
        "mismatch_monitor": "ON",
        "zone_fallback": "READY",
        "recovery_result": scenario["recovery_state"],
    }

    return detections


def apply_mvpv3_state_if_available():
    dashboard_state = load_mvpv3_dashboard_state()
    if dashboard_state is None:
        return None

    detections = dashboard_state.pop("detections", [])
    app_state.update(dashboard_state)
    app_state["integration_source"] = "MVPv3"
    return detections


@router.get("/api/status")
def get_status():
    detections = apply_mvpv3_state_if_available()
    if detections is None:
        build_mock_detections()
        app_state["integration_source"] = "MOCK"
    app_state["command_bridge"] = command_bridge_status()
    return app_state


@router.get("/api/detections")
def get_detections():
    detections = apply_mvpv3_state_if_available()
    if detections is not None:
        return detections
    return build_mock_detections()


@router.post("/api/zone/toggle")
def toggle_zone_lock():
    ensure_base_state()
    app_state["zone_lock"] = "OFF" if app_state["zone_lock"] == "ON" else "ON"
    command = write_command("ZONE_LOCK_TOGGLE", {"zone_lock": app_state["zone_lock"]})
    return {"status": app_state["zone_lock"], "command": command}
