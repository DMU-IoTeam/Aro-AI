# Aro-AI (AI 기반 낙상 감지 시스템)

Aro-AI는 MediaPipe와 GRU(Gated Recurrent Unit)를 활용하여 실시간 영상에서 사람의 자세를 분석하고 낙상 사고를 감지하는 인공지능 프로젝트입니다.

---

## 🚀 주요 기능
- **MediaPipe Pose Extraction**: 영상에서 33개의 인체 관절 좌표를 실시간으로 추출
- **시계열 데이터 분석**: 16프레임 시퀀스 데이터를 GRU 모델로 분석하여 동작의 흐름 파악
- **Focal Loss 적용**: 데이터 비대칭 문제를 해결하여 낙상 탐지 정확도 향상
- **실시간 추론**: 웹캠 영상을 실시간으로 분석하여 낙상 위험 감지 시 경고

## 🛠 기술 스택
- **Language**: Python 3.10+
- **Deep Learning**: PyTorch 2.5
- **Computer Vision**: OpenCV, MediaPipe
- **Data Analysis**: NumPy, Scikit-learn
- **Visualization**: Matplotlib, Seaborn

## 📂 폴더 구조
```text
.
├── dataset/                  # 데이터셋 로더 및 전처리 스크립트
├── model/                    # GRUClassifier 모델 정의
├── utils/                    # Loss 함수, Scheduler, 분석 지표 등 유틸리티
├── visualization/            # 학습 결과 시각화 도구
├── extracted_data/           # 전처리된 .npy 데이터 저장 폴더
├── results/                  # 학습 리포트 및 성능 지표 저장
├── main.py                   # 모델 학습 실행 파일
└── realtime_infer.py         # 실시간 웹캠 추론 스크립트
```

## ⚙️ 설치 및 실행 방법

### 1. 환경 설정 (Conda 사용 권장)
```bash
# 가상환경 생성 및 활성화
conda env create -f environment.yml
conda activate aro-ai

# 또는 requirements.txt를 통한 설치
pip install -r requirements.txt
```

### 2. 데이터 전처리
원본 영상에서 관절 좌표를 추출하여 `.npy` 파일로 변환합니다.
```bash
python extracting_npy.py
```

### 3. 모델 학습
설정된 데이터를 바탕으로 GRU 모델을 학습시킵니다.
```bash
python main.py
```

### 4. 실시간 추론 실행
웹캠을 통해 실시간으로 낙상을 감지합니다.
```bash
python realtime_infer.py
```

---

## 📊 분석 및 결과
- `results/` 폴더 내에서 `training_metrics.png`와 `confusion_matrix.png`를 통해 모델의 학습 성능을 확인할 수 있습니다.
- 데이터 불균형을 고려하여 **Weighted F1-Score**를 주요 평가 지표로 사용합니다.