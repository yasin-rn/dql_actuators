import torch
from data_loader import DatasetLoader
from neural_network import TransformerEncoderNetwork
from simulation_connection import SimulationConnection
from datetime import datetime
import os
import numpy as np

input_headers = [
    "ActuatorDeviations"]
output_headers = ["ActuatorActions"]


loader = DatasetLoader("Dataset_70_210520250036.json")

input_datas, output_datas = loader.get_seq_data(
    5, input_headers, output_headers)

input_tensor = torch.tensor(input_datas, dtype=torch.float32, device="cuda")
output_tensor = torch.tensor(
    output_datas, dtype=torch.float32, device="cuda").squeeze(-1)

input_dim = 48
model_dim = 64
num_heads = 16
num_layers = 32
output_dim = 48*3
learning_rate = 1e-4
sequence_length = 5

model = TransformerEncoderNetwork(
    input_dim=input_dim,
    model_dim=model_dim,
    num_heads=num_heads,
    num_layers=num_layers,
    output_dim=output_dim,
    learning_rate=learning_rate
)

for i in range(1000):
    predicted = model.forward(input_tensor)
    model.backward(predicted, output_tensor)
    print_detail = False
    if i % 100 == 0:
        print_detail = True
        print("Step: ", i, "Loss: ", end="")
    model.update(print_detail)

connection = SimulationConnection()

for i in range(50):
    input_tensor = torch.tensor(
        connection.get_actuator_deviation_ts(), dtype=torch.float32, device="cuda")
    predicted_tensor = model.predict(input_tensor)
    predicted = predicted_tensor.cpu().detach().numpy().tolist()
    connection.set_actuator_action(predicted)
    print(connection.get_sigma_2())


def create_chekpoint_dir():
    time = datetime.now()
    dir_name = time.strftime("%Y-%m-%d_%H-%M-%S")
    full_path = os.path.join("checkpoints", dir_name)
    os.makedirs(full_path, exist_ok=True)
    return full_path
