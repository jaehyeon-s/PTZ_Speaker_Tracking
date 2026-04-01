import socket

# VISCA over IP 기본 포트 (UDP)
UDP_IP = "127.0.0.1"
UDP_PORT = 52381

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

print(f"가상 PTZ 카메라 대기 중... (IP: {UDP_IP}, Port: {UDP_PORT})")

while True:
    data, addr = sock.recvfrom(1024)
    
    if len(data) >= 8:
        payload_type = data[0:2].hex()
        payload_length = int.from_bytes(data[2:4], byteorder='big')
        sequence_number = data[4:8].hex()
        visca_command = data[8:].hex()
        
        print(f"\n[수신됨] 클라이언트: {addr}")
        print(f" - 시퀀스 넘버: {sequence_number}")
        print(f" - VISCA 명령어 (Payload): {visca_command.upper()}")
        
        # 헤더(8바이트) + ACK
        ack_payload = bytes.fromhex("9041FF")
        ack_header = bytes.fromhex("01110003") + data[4:8]
        
        sock.sendto(ack_header + ack_payload, addr)
        print(" -> 가짜 ACK 패킷 전송 완료")
    else:
        print(f"알 수 없는 패킷 수신: {data.hex()}")
