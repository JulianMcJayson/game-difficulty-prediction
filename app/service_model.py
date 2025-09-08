from app.model import train_and_test, GamePredictionModel
from app.api_model import GameRequestModel, GameResponseModel
import joblib
import torch
import json
import numpy as np

DEVICE = "mps" if torch.mps.is_available() else "cpu"
EPOCH = 10

def train_and_test_caller():
    #comment this if you don't want to train again
    with open("data/classification_dataset.json", "r+") as file:
        dataset = json.loads(file.read())
        train_and_test(dataset, EPOCH)

    #test model usage
    model = GamePredictionModel()
    model.load_state_dict(torch.load("model/adaptive_model.pth", weights_only=True, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    x_scalar = joblib.load('model/x_scalar.gz')

    with open("data/normalized_data.json", "r+") as file:
        data = json.loads(file.read())
    for i in data:
        activity = (i["movement_rate"] + i["rotation_rate"] + i["normalized_action_rate"]) / 3 
        x_input = x_scalar.transform([[np.log1p(i["fail"]), activity]])
        x_input_torch = torch.tensor(x_input, dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            output = model(x_input_torch)
            predict = (torch.sigmoid(output) > 0.5).float()
            # prob = torch.softmax(output, dim=1)
            # predict = torch.argmax(output, dim=1)
        fail = i["fail"]
        print(f"Fail: {fail}, Activity: {activity}" )
        print(f"Predict Class : {predict.item()}")


def predict(data: GameRequestModel):
    """
    for api endpoint to predict difficulty
    """
    model = GamePredictionModel()
    model.load_state_dict(torch.load("model/adaptive_model.pth", weights_only=True, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    x_scalar = joblib.load('model/x_scalar.gz')
    activity = (data.movement_rate + data.rotation_rate + data.action_rate) / 3 
    x_input = x_scalar.transform([[np.log1p(data.fail), activity]])
    x_input_torch = torch.tensor(x_input, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        output = model(x_input_torch)
        predict = (torch.sigmoid(output) > 0.5).float()
        return GameResponseModel(adaptive_difficulty=int(predict.item()))