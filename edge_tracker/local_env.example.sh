# Copy this file to local_env.sh and fill in your camera details.
# local_env.sh is gitignored so real credentials never reach the repository.

export CAMERA_HOST="CAMERA_HOST"             # host used for HTTP/Qon control
export CAMERA_IP="CAMERA_IP"                 # LAN IP of the camera
export CAMERA_USER="USER"
export CAMERA_PASSWORD="PASSWORD"

export RTSP_URL="rtsp://${CAMERA_USER}:${CAMERA_PASSWORD}@${CAMERA_IP}:554/stream2"
export QON_CAMERA_URL="http://${CAMERA_IP}"
