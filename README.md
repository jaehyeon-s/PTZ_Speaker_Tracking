# ModelTest Branch

YOLO26n 모델 추론 성능 측정 및 환경 설정 테스트 브랜치입니다.

## 테스트 환경

| 항목 | 사양 |
|---|---|
| 하드웨어 | NVIDIA Jetson Xavier AGX |
| JetPack | 5.1.3 (R35.5.0) |
| Python | 3.8 |
| PyTorch | 2.1.0a0+41361538.nv23.6 |
| Ultralytics | 8.4.26 |
| 카메라 | 한화테크윈 QND-6011 |
| 스트림 | RTSP / H.264 / 640×480 / 30fps |

## 추론 성능 측정 결과

| 방식 | imgsz | 평균 FPS | 비고 |
|---|---|---|---|
| Xaiver CPU | 416 | 3.4 | ARM CPU, 최적화 없음 |
| Xaiver NCNN | 416 | 15.4 | ARM 최적화 |
| Xaiver NCNN + ByteTrack | 416 | 14.3 | - |
| PI5 CPU | 416 | 7.6 | ARM CPU, 최적화 없음 |
| PI5 NCNN | 416 | 27.2 | ARM 최적화 |
| PI5 NCNN + ByteTrack | 416 | 27.2 | - |

## 실행 방법
```bash
# 가상환경 설정 (JetPack 환경)
python3.8 -m venv ~/.venv --system-site-packages
source ~/.venv/bin/activate
pip install ultralytics opencv-python ncnn

# NCNN 변환
python export_ncnn.py

# RTSP 스트림 추론 테스트
python yolo26n.py
python yolo26n_ncnn.py
```

## 파일 구조
```
ModelTest/
├── rtsp_test.py      # RTSP 스트림 + YOLO 추론 테스트
├── export_ncnn.py    # NCNN 변환 스크립트
└── README.md
``` 

## 주요 이슈 및 해결

| 이슈 | 원인 | 해결 |
|---|---|---|
| ncnn import 오류 | 파일명이 라이브러리명과 충돌 | 파일명 변경 |
| libopenblas 오류 | 시스템 라이브러리 누락 | apt install libopenblas-dev |
