# Run this inside the MVPv3 project root after editing paths and camera info.
$env:QON_PASS="카메라비밀번호"

python scripts/mvpv3_command_worker.py `
  --command-path "C:\Users\USER\OneDrive\Desktop\PTZ_Speaker_Tracking\runtime\dashboard_command.json" `
  --ack-path "C:\Users\USER\OneDrive\Desktop\PTZ_Speaker_Tracking\runtime\dashboard_command_ack.json" `
  --camera-url "http://카메라IP" `
  --username "admin" `
  --password-env QON_PASS
