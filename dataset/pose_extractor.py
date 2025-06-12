import cv2
import numpy as np
import mediapipe as mp

class PoseExtractor:
    def __init__(self, num_frames=16):
        self.pose = mp.solutions.pose.Pose(static_image_mode=False)
        self.num_frames = num_frames
        self.prev = None
        self.ESSENTIAL_LANDMARKS = [0, 11, 12, 13, 14, 15, 16,
                                    23, 24, 25, 26, 27, 28]

    def extract_keypoints(self, video_path):
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_idxs = np.linspace(0, total_frames - 1, self.num_frames).astype(int)
        keypoints_sequence = []

        for i in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            if i in frame_idxs:
                frame = cv2.resize(frame, (256, 144))
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = self.pose.process(frame_rgb)

                if result.pose_landmarks:
                    keypoints = []
                    for idx in self.ESSENTIAL_LANDMARKS:
                        lm = result.pose_landmarks.landmark[idx]
                        keypoints.extend([lm.x, lm.y, lm.z, lm.visibility])
                else:
                    keypoints = [0.0] * (len(self.ESSENTIAL_LANDMARKS) * 4)

                keypoints_sequence.append(keypoints)

        cap.release()

        if len(keypoints_sequence) == 0:
            return None

        # 프레임 수 보정
        while len(keypoints_sequence) < self.num_frames:
            keypoints_sequence.append(keypoints_sequence[-1])

        keypoints_sequence = np.array(keypoints_sequence)
        velocity = np.diff(keypoints_sequence, axis=0, prepend=keypoints_sequence[0:1])
        return np.concatenate([keypoints_sequence, velocity], axis=1)

    
    def extract_keypoints_from_frame(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.pose.process(frame_rgb)

        if result.pose_landmarks:
            keypoints = []
            for idx in self.ESSENTIAL_LANDMARKS:
                lm = result.pose_landmarks.landmark[idx]
                keypoints.extend([lm.x, lm.y, lm.z, lm.visibility])
        else:
            keypoints = [0.0] * (len(self.ESSENTIAL_LANDMARKS) * 4)

        keypoints = np.array(keypoints, dtype=np.float32)

        if self.prev is None:
            velocity = np.zeros_like(keypoints)
        else:
            velocity = keypoints - self.prev

        self.prev = keypoints
        return np.concatenate([keypoints, velocity])
