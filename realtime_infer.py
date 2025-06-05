import cv2
import numpy as np
import torch
from model.gru_model import GRUClassifier
from dataset.pose_extractor import PoseExtractor

# 모델 및 추론 파라미터
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GRUClassifier(input_size=264, hidden_size=128, num_layers=1, num_classes=2).to(device)
model.load_state_dict(torch.load("best_model.pt", map_location=device))
model.eval()

extractor = PoseExtractor(num_frames=16)
frame_buffer = []

# 낙상 의심 조건 설정
def is_suspicious_change(curr_kps, prev_kps, threshold=0.3):
    head_idx = 0  # 0번이 nose (머리)
    curr_y = curr_kps[head_idx * 4 + 1]
    prev_y = prev_kps[head_idx * 4 + 1]
    return abs(curr_y - prev_y) > threshold

cap = cv2.VideoCapture(0)

prev_kps = None
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    kp_with_vel = extractor.extract_keypoints_from_frame(frame)
    if kp_with_vel is None:
        continue

    # buffer에 쌓기
    frame_buffer.append(kp_with_vel)

    # 충분히 쌓였을 때만 판단
    if len(frame_buffer) >= 16:
        if prev_kps and is_suspicious_change(kp_with_vel[:132], prev_kps[:132]):
            sequence = np.stack(frame_buffer[-16:], axis=0)
            input_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(device)
            pred = torch.argmax(model(input_tensor), dim=1).item()

            if pred == 1:
                print("⚠️ 낙상 감지됨!")

        prev_kps = kp_with_vel[:132]

cap.release()
cv2.destroyAllWindows()
