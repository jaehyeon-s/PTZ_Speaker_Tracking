# PTZ Speaker Tracking System

**발화자 인식 기반 PTZ 카메라 자동 추적 시스템**

강의실, 회의실 등의 환경에서 발화자를 자동으로 인식하고  
PTZ 카메라를 실시간으로 제어하는 엣지 디바이스 기반 시스템입니다.

---

## 프로젝트 개요

| 항목 | 내용 |
|------|------|
| **기관** | (주)엠디케이 |
| **인원** | 3명 |
| **플랫폼** | Raspberry Pi 5 |
| **카메라 제어** | Sony VISCA over IP |

---

## 주요 기능

- 실시간 발화자 인식 및 자동 추적
- 다중 인물 환경에서의 발화자 판별
- 사용자 지정 화면 구도 제어 (상반신/전신, 좌/중/우)
- 타깃 분실 시 자동 복구
- 웹 기반 대시보드 및 REST API
- Docker 기반 배포

---

## 기술 스택

| 분류 | 기술 |
|------|------|
| AI 추론 | YOLO26n (NCNN) |
| 트래킹 | ByteTrack |
| 카메라 제어 | VISCA over IP |
| API | FastAPI |
| 배포 | Docker |
| 언어 | Python 3.11+ |

---

## 프로젝트 구조

```
PTZ_Speaker_Tracking/
├── src/
│   ├── common/          # 공통 모듈
│   ├── vision/          # 비전 파이프라인
│   ├── audio/           # 오디오 파이프라인
│   ├── fusion/          # 센서 융합
│   ├── ptz/             # PTZ 카메라 제어
│   └── api/             # REST API 및 웹 대시보드
├── tests/               # 테스트
├── configs/             # 설정 파일
├── scripts/             # 유틸리티 스크립트
├── docs/                # 문서
└── docker/              # Docker 관련
```

---

## 빠른 시작

### 요구사항

- Raspberry Pi 5 (8GB)
- RTSP + VISCA over IP 지원 PTZ 카메라
- USB 마이크 어레이
- Python 3.11+

### 설치 및 실행

```bash
git clone https://github.com/jaehyeon-s/PTZ_Speaker_Tracking.git
cd PTZ_Speaker_Tracking
pip install -r requirements.txt

cp configs/default.yaml configs/local.yaml
# local.yaml 수정 후 실행
python main.py --config configs/local.yaml
```

---

## 역할 분담

| 담당 | 영역 |
|------|------|
| 1번 | Vision & Audio |
| 2번 | PTZ 제어 |
| 3번 | 통합 / API / UI |

---

## 참고 자료

- YOLO26 (Ultralytics, 2026)
- *Robust Localization of Multiple Speakers with SRP-PHAT*
- *STNet: Speaker Tracking Network*
- Sony VISCA Command List Version 2.00

---

## License

This project is developed as a graduation project.