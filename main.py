from typing import Union
from fastapi import FastAPI

# model.py를 가져온다.
import model

# 그 안에 있는 AndModel 클래스의 인스턴스를 생성한다.
model = model.AndModel()

# API 서버를 생성한다.
app = FastAPI()

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
@app.get("/predict/left/{left}/right/{right}") 
def predict(left: int, right: int):
    result = model.predict([left, right])
    return {"result": result}

# 모델의 학습을 요청한다. 생성 기능은 POST로 한다.
@app.post("/train")
def train():
    model.train()
    return {"result": "OK"}