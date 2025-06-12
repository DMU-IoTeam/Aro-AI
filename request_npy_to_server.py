import requests
import numpy as np

# 서버 주소 (예: 로컬에서 돌리는 경우)
url = "http://localhost:5000/test"

# 서버에 보낼 데이터 (예: JSON 형식)
npy_path = "reduced_extracted_data/npy/val/fall/00001_H_A_SY_C4.npy"
# npy_path = "reduced_extracted_data/npy/val/normal/00005_H_A_N_C1.npy"
# npy_path = "temp/fall.npy"
# npy_path = "temp/normal.npy"
data_array = np.load(npy_path)  # shape: (16, 104) 예상
# 2. numpy 배열을 list로 변환 (JSON 직렬화 가능하게)
data_list = data_array.tolist()

# 3. 전송할 JSON 데이터 구조
payload = {
    "input": data_list  # 여기서 'input' 키는 서버가 기대하는 key로 맞춰줘야 함
}

# 요청 보내기 (POST)
response = requests.post(url, json=payload)

# 응답 확인
if response.status_code == 200:
    print("✅ 예측 결과:", response.json())
else:
    print("❌ 오류 발생:", response.status_code, response.text)
