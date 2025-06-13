from flask import Flask, request, jsonify
import torch
import numpy as np
from model.gru_model import FallGRUClassifier
from utils.notify import send_fall_alert # 구현해야 함 알림 기능

app = Flask(__name__)

# 모델 로드
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FallGRUClassifier(input_size=104, hidden_size=128, num_layers=1, num_classes=2).to(device)
model.load_state_dict(torch.load("saved_model/reduced_fall_gru_best.pt", map_location=device))
model.eval()

@app.route("/predict", methods=["POST"])
def predict_fall():
    try:
        data = request.json["sequence"]  # (16, 104) 시퀀스
        sequence = torch.tensor(data, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = torch.argmax(model(sequence), dim=1).item()

        if pred == 1:
            send_fall_alert()  # 낙상 알림 전송
            return jsonify({"result": "fall", "alert_sent": True})
        else:
            return jsonify({"result": "normal", "alert_sent": False})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
