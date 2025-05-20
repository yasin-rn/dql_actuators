import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Agent_Intelligence(nn.Module):
    def __init__(self, learning_rate):
        super(Agent_Intelligence, self).__init__()
        self.fc1 = nn.Linear(96, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 144)
        self.optimizer = optim.Adam(self.parameters(), learning_rate)
        self.loss_fn = nn.MSELoss()
        self.loss_list = []

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

    def backward(self, current, target):
        loss = self.loss_fn(current, target)
        loss.backward()
        self.loss_list.append(loss.item())
        pass

    def update(self):
        avg_loss = sum(self.loss_list)/len(self.loss_list)
        print("Loss: ", avg_loss)
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.loss_list.clear()
        pass
