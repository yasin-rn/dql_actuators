import torch
from data_loader import DatasetLoader
from neural_network import NeuralNetwork


loader = DatasetLoader("Dataset_70_200520251831.json")
network = NeuralNetwork(0.001)

input_headers = ["ActuatorPositions", "ActuatorDeviations"]
output_headers = ["ActuatorActions"]
input_data, output_data = loader.get_nn_data(1, input_headers, output_headers)

input_tensors = torch.from_numpy(input_data)
output_tensors = torch.from_numpy(output_data)
nn_out = network.forward(input_tensors)

network.backward(nn_out, output_tensors)
