import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from model.gru_model import FallGRUClassifier
from config import CFG


# ✅ 단일 npy 추론 함수
def infer_npy_file(model, npy_path, device):
    try:
        keypoints = np.load(npy_path)  # (num_frames, feat_dim)
        x = torch.tensor(keypoints, dtype=torch.float32).unsqueeze(0).to(device)
    except:
        return None, None

    with torch.no_grad():
        out = model(x)
        prob = torch.softmax(out, dim=1).cpu().numpy()[0]
        return np.argmax(prob), prob


# ✅ 디렉토리 전체 추론 (npy 기준)
def run_inference_on_directory(model, root_dir, device):
    results = []
    label_map = {"normal": "Normal", "fall": "Fall"}

    for class_folder in ["normal", "fall"]:
        class_path = os.path.join(root_dir, class_folder)
        for root, dirs, files in os.walk(class_path):
            for file in files:
                if not file.endswith(".npy"):
                    continue
                npy_path = os.path.join(root, file)
                pred, prob = infer_npy_file(model, npy_path, device)
                if pred is None:
                    continue
                results.append({
                    "true": label_map[class_folder],
                    "pred": "Fall" if pred == 1 else "Normal",
                    "path": npy_path
                })
    return pd.DataFrame(results)


# ✅ 분석 함수 (시점/Cx, fall type 추출 가능 시 사용)
def get_fall_type(path):
    if "FY" in path:
        return "FY"
    elif "BY" in path:
        return "BY"
    elif "SY" in path:
        return "SY"
    return "Unknown"

def get_camera_view(path):
    for c in range(1, 9):
        if f"C{c}" in path:
            return f"C{c}"
    return "Unknown"

def analyze_results(df):
    df["fall_type"] = df["path"].apply(get_fall_type)
    df["camera"] = df["path"].apply(get_camera_view)
    df["correct"] = df["true"] == df["pred"]

    print("\n📊 FY/BY/SY 오분류 (Fall인데 Normal로 분류된 경우):")
    print(df[(df["true"] == "Fall") & (df["pred"] == "Normal")]["fall_type"].value_counts())

    print("\n📊 시점별 정확도:")
    print(df.groupby("camera")["correct"].mean().sort_index())

    df.groupby("camera")["correct"].mean().sort_index().plot(kind="bar", ylim=(0, 1), title="시점별 정확도")
    plt.ylabel("Accuracy")
    plt.xlabel("Camera View")
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.show()


# ✅ 실행 진입점
if __name__ == "__main__":
    model = FallGRUClassifier()
    model.load_state_dict(torch.load(CFG.save_path, map_location=CFG.device))
    model.to(CFG.device)
    model.eval()

    os.makedirs("results", exist_ok=True)

    # ✅ 추론 실행 (npy 기준)
    df_result = run_inference_on_directory(model, root_dir="extracted_data/npy/val", device=CFG.device)

    # ✅ 저장 및 분석
    df_result.to_csv("results/inference_results.csv", index=False)
    analyze_results(df_result)
