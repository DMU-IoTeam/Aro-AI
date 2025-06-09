import cv2
import numpy as np
import torch
import requests
import time
from dataset.pose_extractor import PoseExtractor  # 동일한 모듈 사용
import json
import os

# 버퍼 설정
NUM_FRAMES = 16
frame_buffer = []
prev_kps = None

# PoseExtractor
extractor = PoseExtractor(num_frames=NUM_FRAMES)

# 낙상 의심 조건: 머리 좌표 y 변화량으로 간단 정의
def is_suspicious_change(curr_kps, prev_kps, threshold=0.3):
    head_idx = 0
    curr_y = curr_kps[head_idx * 4 + 1]
    prev_y = prev_kps[head_idx * 4 + 1]
    return abs(curr_y - prev_y) > threshold

# 서버 주소
SERVER_URL = "http://<YOUR_AI_SERVER_IP>:<PORT>/inference"  # 예: http://192.168.0.10:5000/inference

# 웹캠 연결
cap = cv2.VideoCapture(0)
frame_count = 0

print("✅ 낙상 감지 시작 (라즈베리파이)")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 관절 추출
    kp_with_vel = extractor.extract_keypoints_from_frame(frame)
    if kp_with_vel is None:
        continue

    frame_buffer.append(kp_with_vel)

    # 버퍼 유지
    if len(frame_buffer) > NUM_FRAMES:
        frame_buffer.pop(0)

    # 추론 조건
    if prev_kps is not None and is_suspicious_change(kp_with_vel[:52], prev_kps[:52]):
        if len(frame_buffer) == NUM_FRAMES:
            print("⚠️ 낙상 의심 상황 발생 - 서버 전송 중...")

            # npy 임시 저장
            npy_data = np.stack(frame_buffer)
            tmp_path = "/tmp/sequence.npy"
            np.save(tmp_path, npy_data)

            # 서버로 전송
            with open(tmp_path, 'rb') as f:
                files = {'npy': f}
                try:
                    response = requests.post(SERVER_URL, files=files, timeout=5)
                    print("📩 서버 응답:", response.text)
                except Exception as e:
                    print(f"[❗] 서버 전송 실패: {e}")

            # 중복 전송 방지를 위해 약간의 시간 지연
            time.sleep(3)

    prev_kps = kp_with_vel[:52]
    frame_count += 1

cap.release()
cv2.destroyAllWindows()
print("🛑 종료됨")
