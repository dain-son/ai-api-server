from typing import Union
from fastapi import FastAPI

# model.py를 가져온다.
import model

# 그 안에 있는 AndModel 클래스의 인스턴스를 생성한다.
gate_models = {
    'AND': model.GateModel('AND'),
    'OR': model.GateModel('OR'),
    'NOT': model.GateModel('NOT')
}
# API 서버를 생성한다.
app = FastAPI()

# 모델의 학습을 요청한다. 생성 기능은 POST로 한다.
@app.post("/train/{gate_type}")
def train(gate_type: str):
    gate_type = gate_type.upper()
    if gate_type not in gate_models:
        return {"error": "지원하지 않는 gate_type입니다. AND, OR, NOT 중 선택해주세요."}
    gate_models[gate_type].train()
    return {"result": f"{gate_type} 게이트 학습 완료!"}

# endpoint 엔드포인트를 선언하며 GET으로 요청을 받고 경로는 /이다.
@app.get("/")
def read_root():
    # 딕셔너리를 반환하면 JSON으로 직렬화된다.
    return {"Hello": "World"}

# 이 엔드포인트의 전체 경로는 /items/{item_id} 이다.
# 중괄호안의 item_id는 경로 매개변수(파라메터)이며 데코레이터 아래 함수의 인수로 쓰인다.
@app.get("/items/{item_id}") 
def read_item(item_id: int):
    return {"item_id": item_id}

# 모델의 예측 기능을 호출한다. 조회 기능은 GET로 한다.
@app.get("/predict/{gate_type}/{left}/{right}") 
def predict(gate_type: str, left: int, right: int):
    gate_type = gate_type.upper()
    if gate_type not in gate_models:
        return {"error": "지원하지 않는 gate_type입니다. AND, OR, NOT 중 선택해주세요."}
    
    result = gate_models[gate_type].predict([left, right])
    return {"result": result}


@app.get("/predict/NOT/{input_value}")
def predict_not(input_value: int):
    result = gate_models['NOT'].predict([input_value])
    return {"result": result}

import pickle 

# 모델 객체 저장
with open("model.pkl", "wb") as file:
    pickle.dump(gate_models, file)

