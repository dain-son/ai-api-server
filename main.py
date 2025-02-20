from typing import Union
from fastapi import FastAPI

# model.py를 가져온다.
import model
import pickle 

# API 서버를 생성한다.
app = FastAPI()

def load_model(gate_type: str):
    with open(f"./{gate_type}_model.pkl", "rb") as f:
        return pickle.load(f)


# 모델의 예측 기능을 호출한다. 조회 기능은 GET로 한다.
@app.get("/predict/{gate_type}/{left}/{right}") 
def predict(gate_type: str, left: int, right: int):
    gate_type = gate_type.upper()

    # 지원하지 않는 gate_type이 요청되면 에러 반환
    if gate_type not in ["AND", "OR", "NOT"]:
        return {"error": "지원하지 않는 gate_type입니다. AND, OR, NOT 중 선택해주세요."}
    
    gate_model = load_model(gate_type)
    result = gate_model.predict([left, right])
    return {"result": result}



