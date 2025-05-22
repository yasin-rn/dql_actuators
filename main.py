import torch
from data_loader import DatasetLoader
from neural_network import TransformerEncoderNetwork

input_headers = [
    "ActuatorPositions"]
output_headers = ["ActuatorActions"]


loader = DatasetLoader("Dataset_70_200520251831.json")

input_datas, output_datas = loader.get_seq_data(
    3, input_headers, output_headers)

input_tensor = torch.tensor(input_datas, dtype=torch.float32)
output_tensor = torch.tensor(output_datas)

# Model parametreleri
input_dim = 48
model_dim = 64
num_heads = 8
num_layers = 2
output_dim = 48
learning_rate = 1e-4
sequence_length = 3
batch_size = 17

# Transformer modeli oluştur
model = TransformerEncoderNetwork(
    input_dim=input_dim,
    model_dim=model_dim,
    num_heads=num_heads,
    num_layers=num_layers,
    output_dim=output_dim,
    learning_rate=learning_rate
)
model.forward(input_tensor)
