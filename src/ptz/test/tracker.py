import socket

class PTZTracker:
    def __init__(self, ip='127.0.0.1', port=52381):
        self.camera_ip = ip
        self.camera_port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sequence_number = 1
        
        self.PAN_MAX_SPEED = 24  
        self.TILT_MAX_SPEED = 20 

    def send_tracking_cmd(self, img_width, img_height, bbox_center_x, bbox_center_y):
        center_x = img_width / 2
        center_y = img_height / 2
        
        error_x = bbox_center_x - center_x
        error_y = bbox_center_y - center_y
        
        DEADZONE_X = img_width * 0.1
        DEADZONE_Y = img_height * 0.1
        
        KP_X = self.PAN_MAX_SPEED / (img_width / 2)
        KP_Y = self.TILT_MAX_SPEED / (img_height / 2)
        
        pan_speed = 1
        pan_dir = 0x03 
        if abs(error_x) > DEADZONE_X:
            raw_speed = int(abs(error_x) * KP_X)
            pan_speed = max(1, min(raw_speed, self.PAN_MAX_SPEED))
            pan_dir = 0x02 if error_x > 0 else 0x01 
            
        tilt_speed = 1
        tilt_dir = 0x03 
        if abs(error_y) > DEADZONE_Y:
            raw_speed = int(abs(error_y) * KP_Y)
            tilt_speed = max(1, min(raw_speed, self.TILT_MAX_SPEED))
            tilt_dir = 0x02 if error_y > 0 else 0x01 

        visca_payload = bytes([0x81, 0x01, 0x06, 0x01, pan_speed, tilt_speed, pan_dir, tilt_dir, 0xFF])
        
        header = bytes([0x01, 0x00, 0x00, 0x09]) + self.sequence_number.to_bytes(4, byteorder='big')
        
        packet = header + visca_payload
        self.sock.sendto(packet, (self.camera_ip, self.camera_port))
        
        print(f"[Seq:{self.sequence_number:04d}] Offset(X:{error_x:>5.1f}, Y:{error_y:>5.1f}) -> Packet: ...")
        
        self.sequence_number += 1
        if self.sequence_number > 0xFFFFFFFF:
            self.sequence_number = 1
