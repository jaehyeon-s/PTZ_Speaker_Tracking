import time
from tracker import PTZTracker

def run_test_scenario():
    tracker = PTZTracker(ip='127.0.0.1', port=52381)
    
    width, height = 1920, 1080
    
    print("VISCA IP 통신 테스트")
    
    # test_1
    print("\n오른쪽으로 이동 명령")
    tracker.send_tracking_cmd(width, height, 1500, 540)
    time.sleep(1)
    
    # test_2
    print("\n왼쪽으로 이동 명령")
    tracker.send_tracking_cmd(width, height, 400, 200)
    time.sleep(1)
    
    # test_3
    print("\n정지 명령")
    tracker.send_tracking_cmd(width, height, 980, 550)
    time.sleep(1)
    
    print("\ntest complete")

if __name__ == "__main__":
    run_test_scenario()
