import functools
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import colossalai
from colossalai.booster import Booster
import torch.distributed as dist
import sys
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
print = functools.partial(print, flush=True)


class MLPLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.ReLU(),
            nn.BatchNorm1d(output_size),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        return self.layer(x)


class MLP(nn.Module):
    def __init__(self, input_size=784, hidden_sizes=[512, 256], output_size=10):
        super().__init__()
        self.input_size = input_size

        # Create pipeline-friendly structure
        self.layers = nn.ModuleList([
            MLPLayer(input_size, hidden_sizes[0]),
            MLPLayer(hidden_sizes[0], hidden_sizes[1]),
            # MLPLayer(hidden_sizes[1], hidden_sizes[2]),
            # MLPLayer(hidden_sizes[2],hidden_sizes[3]),
            nn.Linear(hidden_sizes[1], output_size)
        ])

    def forward(self, x):
        x = x.view(-1, self.input_size)
        for layer in self.layers:
            x = layer(x)
        return x


def setup_dataloader(batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    dataset = datasets.MNIST('./data', train=True,
                             download=True, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def train_regular(num_epochs=1, batch_size=32):
    device = torch.device("cuda:0")
    print(f"Using Device: {device}")

    model = MLP().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_dataloader = setup_dataloader(batch_size)

    start_time = time.time()

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for batch_idx, (data, target) in enumerate(train_dataloader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if batch_idx % 100 == 99:
                avg_loss = running_loss / 100
                print(
                    f'Epoch: {epoch + 1}, Batch: {batch_idx + 1}, Loss: {avg_loss:.3f}')
                running_loss = 0.0

    end_time = time.time()
    return end_time - start_time


def main():
    try:
        print("Starting Regular Training...")
        regular_time = train_regular(3, 32)

        print(f"\nRegular Time: {regular_time:.2f} seconds")

    except Exception as e:
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()
