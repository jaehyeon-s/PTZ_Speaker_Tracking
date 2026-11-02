"""Copy this file to local_settings.py and fill in the local camera URLs."""

RTSP_URL = "rtsp://USER:PASSWORD@HOST:554/STREAM"

# Optional. If omitted, qon-http mode derives http://HOST from RTSP_URL.
# You can also use MARKER_TRACKER_QON_CAMERA_URL instead.
QON_CAMERA_URL = ""
