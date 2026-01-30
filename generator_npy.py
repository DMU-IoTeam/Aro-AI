# generate_npy.py

import os, cv2, glob
import numpy as np
import mediapipe as mp
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import random

# ✅ 2. Pose Extractor
# class PoseExtractor:
#     def __init__(self, num_frames=16):
#         self.num_frames = num_frames

#     def extract_keypoints(self, video_path):
#         cap = cv2.VideoCapture(video_path)
#         total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#         frame_idxs = np.linspace(0, total_frames - 1, self.num_frames).astype(int)
#         keypoints_sequence = []

#         with mp.solutions.pose.Pose(static_image_mode=False) as pose:
#             for idx in frame_idxs:
#                 cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
#                 ret, frame = cap.read()
#                 if not ret:
#                     continue
#                 frame = cv2.resize(frame, (256, 144))
#                 frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#                 result = pose.process(frame_rgb)

#                 if result.pose_landmarks:
#                     keypoints = []
#                     for lm in result.pose_landmarks.landmark:
#                         keypoints.extend([lm.x, lm.y, lm.z, lm.visibility])
#                 else:
#                     keypoints = [0] * (33 * 4)

#                 keypoints_sequence.append(keypoints)

#         cap.release()

#         if len(keypoints_sequence) == 0:
#             print(f"[❗] No keypoints extracted in: {video_path}")
#             return None

#         while len(keypoints_sequence) < self.num_frames:
#             keypoints_sequence.append(keypoints_sequence[-1])

#         keypoints_sequence = np.array(keypoints_sequence)
#         velocity = np.diff(keypoints_sequence, axis=0, prepend=keypoints_sequence[0:1])
#         combined = np.concatenate([keypoints_sequence, velocity], axis=1)
#         return combined
class PoseExtractor:
    def __init__(self, num_frames=16):
        self.num_frames = num_frames
        self.important_landmarks = [0, 11, 12, 23, 24, 25, 26, 27, 28]  # 낙상 관련 관절만 사용

    def extract_keypoints(self, video_path):
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_idxs = np.linspace(0, total_frames - 1, self.num_frames).astype(int)
        keypoints_sequence = []

        with mp.solutions.pose.Pose(static_image_mode=False) as pose:
            for idx in frame_idxs:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret:
                    continue
                frame = cv2.resize(frame, (256, 144))
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = pose.process(frame_rgb)

                if result.pose_landmarks:
                    keypoints = []
                    for i in self.important_landmarks:
                        lm = result.pose_landmarks.landmark[i]
                        keypoints.extend([lm.x, lm.y, lm.z, lm.visibility])
                else:
                    keypoints = [0] * (len(self.important_landmarks) * 4)

                keypoints_sequence.append(keypoints)

        cap.release()

        if len(keypoints_sequence) == 0:
            print(f"[❗] No keypoints extracted in: {video_path}")
            return None

        while len(keypoints_sequence) < self.num_frames:
            keypoints_sequence.append(keypoints_sequence[-1])

        keypoints_sequence = np.array(keypoints_sequence)
        velocity = np.diff(keypoints_sequence, axis=0, prepend=keypoints_sequence[0:1])
        combined = np.concatenate([keypoints_sequence, velocity], axis=1)
        return combined

def get_npy_path_from_video_path(video_path):
    fname = os.path.splitext(os.path.basename(video_path))[0]
    return os.path.join("npy_data_16frames", f"{fname}.npy")

def process_video(video_path):
    npy_path = get_npy_path_from_video_path(video_path)
    if os.path.exists(npy_path):
        return npy_path  # 이미 처리된 경우 스킵

    extractor = PoseExtractor(num_frames=16)
    keypoints = extractor.extract_keypoints(video_path)

    if keypoints is not None:
        np.save(npy_path, keypoints)
        return npy_path
    else:
        print(f"[⚠️] 실패: {video_path}")
        return None

if __name__ == "__main__":
    os.makedirs("npy_data_16frames", exist_ok=True)

    fall_videos = glob.glob("C:/videoData/Training/*/*/Y/*/*/*.mp4", recursive=True)
    not_fall_videos = glob.glob("C:/videoData/Training/*/*/N/N/*/*.mp4", recursive=True)
    
        # ✅ 1/4 샘플링 (클래스 비율 유지)
    fall_sample_size = len(fall_videos) // 4
    not_fall_sample_size = len(not_fall_videos) // 4

    fall_sampled = random.sample(fall_videos, fall_sample_size)
    not_fall_sampled = random.sample(not_fall_videos, not_fall_sample_size)

    sampled_video_paths = fall_sampled + not_fall_sampled
    random.shuffle(sampled_video_paths)

    # ✅ 병렬 처리
    with ProcessPoolExecutor(max_workers=4) as executor:  # 필요시 코어 수에 맞게 조절
        results = list(tqdm(executor.map(process_video, sampled_video_paths), total=len(sampled_video_paths)))


    print(f"[✅] 완료: {len([r for r in results if r is not None])}개 처리됨.")