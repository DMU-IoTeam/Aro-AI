import os
import cv2
import numpy as np
import torch
import requests
import mediapipe as mp

from model.gru_model import FallGRUClassifier
from dataset.pose_extractor import PoseExtractor

# ============================ 설정 ============================
# 서버 주소 (Flask 서버와 연동 시 사용)
API_URL = "http://localhost:5000/inference"

# 모델 설정
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL = FallGRUClassifier(input_size=104, hidden_size=128, num_layers=1, num_classes=2).to(DEVICE)
MODEL.load_state_dict(torch.load("reduced_fall_gru_best.pt", map_location=DEVICE))
MODEL.eval()

# PoseExtractor 및 Mediapipe 설정
extractor = PoseExtractor(num_frames=16)
frame_buffer = []
raw_frame_buffer = []

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose_vis = mp_pose.Pose(static_image_mode=False)

# 조정 가능한 파라미터
FRAME_SAMPLE_INTERVAL = 3     # 프레임 샘플링 간격
INFERENCE_INTERVAL = 30       # 추론 간격 (프레임 수 기준)

# ============================ 헬퍼 함수 ============================
def is_suspicious_change(curr_kps, prev_kps, threshold=0.3):
    head_idx = 0
    curr_y = curr_kps[head_idx * 4 + 1]
    prev_y = prev_kps[head_idx * 4 + 1]
    return abs(curr_y - prev_y) > threshold

# ============================ 실시간 처리 ============================
cap = cv2.VideoCapture(0)
frame_count = 0
fall_detected = False
prev_kps = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    vis_frame = frame.copy()
    result = pose_vis.process(cv2.cvtColor(vis_frame, cv2.COLOR_BGR2RGB))
    if result.pose_landmarks:
        mp_drawing.draw_landmarks(vis_frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    if frame_count % FRAME_SAMPLE_INTERVAL == 0:
        kp_with_vel = extractor.extract_keypoints_from_frame(frame)
        if kp_with_vel is None:
            frame_count += 1
            continue

        frame_buffer.append(kp_with_vel)
        raw_frame_buffer.append(vis_frame)

        if len(frame_buffer) >= 16 and not fall_detected and frame_count % INFERENCE_INTERVAL == 0:
            sequence = np.stack(frame_buffer[-16:], axis=0)
            input_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            pred = torch.argmax(MODEL(input_tensor), dim=1).item()

            # ✅ 로컬 추론 결과
            print(f"[로컬] 예측 결과: {pred}")

            # ✅ 서버로도 전송해서 결과 받아오기
            try:
                payload = {"input": sequence.tolist()}
                response = requests.post(API_URL, json=payload)
                if response.status_code == 200:
                    print("[서버] 예측 결과:", response.json())
                else:
                    print("❌ 서버 오류:", response.status_code, response.text)
            except Exception as e:
                print("❌ 서버 요청 실패:", e)

            if pred == 1:
                print(f"⚠️ 낙상 감지됨! (프레임: {frame_count})")
                fall_detected = True

                save_dir = f"fall_detected_frames/detect_{frame_count}"
                os.makedirs(save_dir, exist_ok=True)
                for i in range(16):
                    idx = frame_count - 15 + i
                    if 0 <= idx < len(raw_frame_buffer):
                        cv2.imwrite(f"{save_dir}/frame_{i:02d}.jpg", raw_frame_buffer[idx])

            prev_kps = kp_with_vel[:52]

    resized_frame = cv2.resize(vis_frame, (640, 480))
    cv2.imshow('Aro-AI Fall Detection', resized_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    frame_count += 1

cap.release()
cv2.destroyAllWindows()
print("✅ 실시간 추론 종료")
