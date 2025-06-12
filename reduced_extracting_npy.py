
# ✅ 6. Prepare Data
import os, cv2, glob
import numpy as np
import mediapipe as mp
from tqdm import tqdm


# ✅ 1. 저장 폴더 생성
os.makedirs("reduced_extracted_data/npy/train/fall", exist_ok=True)
os.makedirs("reduced_extracted_data/npy/train/normal", exist_ok=True)
os.makedirs("reduced_extracted_data/npy/val/fall", exist_ok=True)
os.makedirs("reduced_extracted_data/npy/val/normal", exist_ok=True)

# ✅ 2. 영상 경로 수집
train_fall_videos = glob.glob("extracted_video/Training/Y/*/**/*.mp4")
train_normal_videos = glob.glob("extracted_video/Training/N/N/**/*.mp4")
val_fall_videos = glob.glob("extracted_video/Validation/Y/*/**/*.mp4")
val_normal_videos = glob.glob("extracted_video/Validation/N/N/**/*.mp4")

# ✅ 3. 저장 경로 함수
def get_npy_save_path(video_path, label_type, dataset_type):
    fname = os.path.splitext(os.path.basename(video_path))[0]
    return os.path.join(f"reduced_extracted_data/npy/{dataset_type}/{label_type}", f"{fname}.npy")

# ✅ 2. Pose Extractor
class PoseExtractor:
    def __init__(self, num_frames=16):
        self.pose = mp.solutions.pose.Pose(static_image_mode=False)
        self.num_frames = num_frames

        # 🔧 필수 관절 인덱스 선언 (예: 14개 관절)
        self.ESSENTIAL_LANDMARKS = [0, 11, 12, 13, 14, 15, 16,
                                    23, 24, 25, 26, 27, 28]

    def extract_keypoints(self, video_path):
        print("!@#!#!@#!@#!@#!@#")
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_idxs = np.linspace(0, total_frames - 1, self.num_frames).astype(int)
        keypoints_sequence = []

        for i in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            if i in frame_idxs:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = self.pose.process(frame_rgb)

                if result.pose_landmarks:
                    keypoints = []
                    for idx in self.ESSENTIAL_LANDMARKS:
                        lm = result.pose_landmarks.landmark[idx]
                        keypoints.extend([lm.x, lm.y, lm.z, lm.visibility])
                else:
                    keypoints = [0] * (len(self.ESSENTIAL_LANDMARKS) * 4)

                keypoints_sequence.append(keypoints)

        cap.release()

        if len(keypoints_sequence) == 0:
            print(f"[❗] No keypoints extracted in: {video_path}")
            return None

        while len(keypoints_sequence) < self.num_frames:
            keypoints_sequence.append(keypoints_sequence[-1])

        print("#####################")
        keypoints_sequence = np.array(keypoints_sequence)
        velocity = np.diff(keypoints_sequence, axis=0, prepend=keypoints_sequence[0:1])
        combined = np.concatenate([keypoints_sequence, velocity], axis=1)
        print(combined)
        return combined




# ✅ 4. 저장 함수
def process_and_save(videos, label_type, dataset_type):
    extractor = PoseExtractor(num_frames=16)
    success, fail = 0, 0

    for video_path in tqdm(videos, desc=f"{dataset_type}/{label_type}"):
        print("📂 현재 비디오 경로:", video_path)
        print("🔍 파일 존재 여부:", os.path.exists(video_path))
        npy_path = os.path.join(os.path.dirname(__file__), 'temp', f"{label_type}")

        # if os.path.exists(npy_path):
        #     continue

        print("1")
        keypoints = extractor.extract_keypoints(video_path)
        print("2")
        if keypoints is not None:
            # 🔧 디렉터리 자동 생성
            print("#######")
            print(npy_path)
            print("#######")
            os.makedirs(os.path.dirname(npy_path), exist_ok=True)

            np.save(npy_path, keypoints)
            success += 1
        else:
            print(f"[⚠️] 좌표 추출 실패: {video_path}")
            fail += 1

    print(f"✅ 저장 완료 - {dataset_type}/{label_type} | 성공: {success}, 실패: {fail}")


# ✅ 6. 네 종류의 데이터 각각 저장
# process_and_save(train_fall_videos, "fall", "train")
# process_and_save(train_normal_videos, "normal", "train")
# process_and_save(val_fall_videos, "fall", "val")
# process_and_save(val_normal_videos, "normal", "val")
process_and_save(["temp/fall.mp4"], "fall", "val")
process_and_save(["temp/normal.mp4"], "normal", "val")