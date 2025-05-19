import torch
from dataset.pose_extractor import PoseExtractor
import numpy as np

def infer_video_file(model, video_path, device, num_frames=16):
    extractor = PoseExtractor(num_frames)
    keypoints = extractor.extract_keypoints(video_path)
    if keypoints is None:
        return None, None
    x = torch.tensor(keypoints, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(x)
        prob = torch.softmax(out, dim=1).cpu().numpy()[0]
        return np.argmax(prob), prob