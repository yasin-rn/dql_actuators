import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class TransformerEncoderNetwork(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, output_dim, learning_rate):
        super(TransformerEncoderNetwork, self).__init__()

        self.embedding = nn.Linear(input_dim, model_dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim,
                                                   nhead=num_heads,
                                                   dim_feedforward=4*model_dim,
                                                   dropout=0.1,
                                                   activation='relu',
                                                   batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers)

        self.output_layer = nn.Linear(model_dim, output_dim)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        self.loss_list = []

    def decode_output(self, output_tensor):
        reshaped = output_tensor.view(-1, 3)
        predicted_classes = torch.argmax(reshaped, dim=1)
        return predicted_classes

    def forward(self, x):

        x = self.embedding(x)
        x = x.unsqueeze(1)
        x = self.transformer_encoder(x)
        x = x.squeeze(1)
        output = self.output_layer(x)
        return output

    def predict(self, x):
        output = self.forward(x)
        decoded = self.decode_output(output)
        return decoded

    def backward(self, current, target):
        loss = self.loss_fn(current, target)
        loss.backward()
        self.loss_list.append(loss.item())

    def update(self):
        avg_loss = sum(self.loss_list)/len(self.loss_list)
        print("Loss: ", avg_loss)
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.loss_list.clear()
