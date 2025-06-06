import cv2
import numpy as np
import torch
import os
from model.gru_model import FallGRUClassifier
from dataset.pose_extractor import PoseExtractor
import mediapipe as mp

# 모델 및 장치 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FallGRUClassifier(input_size=104, hidden_size=128, num_layers=1, num_classes=2).to(device)
model.load_state_dict(torch.load("reduced_fall_gru_best.pt", map_location=device))
model.eval()

# PoseExtractor 인스턴스 생성
extractor = PoseExtractor(num_frames=16)
extractor.prev = None
frame_buffer = []
raw_frame_buffer = []

# mediapipe skeleton 시각화용
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose_vis = mp_pose.Pose(static_image_mode=False)

# 낙상 의심 조건 정의
def is_suspicious_change(curr_kps, prev_kps, threshold=0.3):
    head_idx = 0
    curr_y = curr_kps[head_idx * 4 + 1]
    prev_y = prev_kps[head_idx * 4 + 1]
    return abs(curr_y - prev_y) > threshold

# 테스트 영상
# cap = cv2.VideoCapture("extracted_video/Validation/Y/BY/00074_H_A_BY_C4/00074_H_A_BY_C4.mp4")
cap = cv2.VideoCapture(0)

prev_kps = None
frame_count = 0
fall_detected = False  # 낙상 중복 감지 방지

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # mediapipe 스켈레톤 시각화
    vis_frame = frame.copy()
    result = pose_vis.process(cv2.cvtColor(vis_frame, cv2.COLOR_BGR2RGB))
    if result.pose_landmarks:
        mp_drawing.draw_landmarks(vis_frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    raw_frame_buffer.append(vis_frame)

    # 관절 + 속도 추출
    kp_with_vel = extractor.extract_keypoints_from_frame(frame)
    if kp_with_vel is None:
        frame_count += 1
        continue

    frame_buffer.append(kp_with_vel)

    if len(frame_buffer) >= 16 and not fall_detected:
        if prev_kps is not None and is_suspicious_change(kp_with_vel[:52], prev_kps[:52]):
            sequence = np.stack(frame_buffer[-16:], axis=0)
            input_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(device)
            pred = torch.argmax(model(input_tensor), dim=1).item()

            if pred == 1:
                print(f"⚠️ 낙상 감지됨! (프레임 번호: {frame_count})")
                fall_detected = True

                save_dir = f"fall_detected_frames/detect_1"
                os.makedirs(save_dir, exist_ok=True)

                for i in range(16):
                    idx = frame_count - 15 + i
                    if 0 <= idx < len(raw_frame_buffer):
                        cv2.imwrite(f"{save_dir}/frame_{i:02d}.jpg", raw_frame_buffer[idx])

        prev_kps = kp_with_vel[:52]

    frame_count += 1

cap.release()
cv2.destroyAllWindows()
print("✅ 추론 및 시각화 완료")
