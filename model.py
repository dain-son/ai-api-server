import numpy as np

class GateModel:
    def __init__(self, gate_type="AND"):
        # 파라메터
        self.gate_type = gate_type.upper()
        self.weights = np.random.rand(2)
        self.bias = np.random.rand(1)

        if self.gate_type == "NOT":
            self.weights = np.random.rand(1) # NOT은 입력이 1개

    def train(self):
        learning_rate = 0.1
        epochs = 20
        if self.gate_type == 'AND':
            inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
            outputs = np.array([0, 0, 0, 1])
        elif self.gate_type == 'OR':
            inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
            outputs = np.array([0, 1, 1, 1])
        elif self.gate_type == 'NOT':
            inputs = np.array([[0], [1]])
            outputs = np.array([1, 0])
        else:
            raise ValueError("지원하지 않는 gate_type 입니다. AND, OR, NOT 중 선택해주세요.")
        
        
        for epoch in range(epochs):
            for i in range(len(inputs)):
                # 총 입력 계산
                total_input = np.dot(inputs[i], self.weights) + self.bias
                # 예측 출력 계산
                prediction = self.step_function(total_input)
                # 오차 계산
                error = outputs[i] - prediction
                print(f'inputs[i] : {inputs[i]}')
                print(f'weights : {self.weights}')
                print(f'bias before update: {self.bias}')
                print(f'prediction: {prediction}')
                print(f'error: {error}')
                # 가중치와 편향 업데이트
                self.weights += learning_rate * error * inputs[i]
                self.bias += learning_rate * error
                print('====')        

    def step_function(self, x):
        return 1 if x >= 0 else 0
    
    def predict(self, input_data):
        input_data = np.array(input_data)
        total_input = np.dot(input_data, self.weights) + self.bias
        return self.step_function(total_input)    
        
    