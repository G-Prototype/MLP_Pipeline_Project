import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.distributed as dist
import time


class MNISTBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)

    def forward(self, x):
        x = self.lin(x)
        x = self.bn(x)
        x = torch.relu(x)
        return x


class ModelChunk0(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layer0 = MNISTBlock(784, 512)

    def forward(self, x):
        x = self.flatten(x)
        return self.layer0(x)


class ModelChunk1(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = MNISTBlock(512, 256)
        self.output = nn.Linear(256, 10)

    def forward(self, x):
        x = self.layer1(x)
        return self.output(x)


class PipelineParallel:
    def __init__(self, model, rank, world_size, device, chunks):
        self.model = model
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.chunks = chunks

    def forward(self, data=None):
        if self.rank == 0:
            micro_batches = data.chunk(self.chunks)
            outputs = []

            for micro_batch in micro_batches:
                output = self.model(micro_batch)
                dist.send(output, dst=self.rank + 1)
                outputs.append(output)

            return torch.cat(outputs)

        elif self.rank == 1:
            outputs = []

            for _ in range(self.chunks):
                received = torch.zeros(
                    data.shape[0] // self.chunks, 512, device=self.device)
                dist.recv(received, src=self.rank - 1)

                output = self.model(received)
                outputs.append(output)

            return torch.cat(outputs)


def train_epoch(rank, world_size, pipeline, train_loader, optimizer, criterion, device):
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()

        data = data.to(device)
        target = target.to(device)

        if rank == 0:
            pipeline.forward(data)
        else:
            output = pipeline.forward(data)
            loss = criterion(output, target)
            loss.backward()

            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            running_loss += loss.item()

            if batch_idx % 100 == 99:
                accuracy = 100. * correct / total
                avg_loss = running_loss / 100
                print(
                    f'Batch: {batch_idx + 1}, Loss: {avg_loss:.3f}, Accuracy: {accuracy:.2f}%')
                running_loss = 0.0
                correct = 0
                total = 0

        optimizer.step()

    dist.barrier()


def main():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo",
                            rank=rank, world_size=world_size)

    device = torch.device(
        f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
    print(f"device={device}")

    torch.manual_seed(42)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(
        root='./data',
        train=True,
        transform=transform,
        download=True
    )

    batch_size = 32
    chunks = 8

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    models = {
        0: ModelChunk0(),
        1: ModelChunk1(),
    }

    if rank in models:
        model = models[rank].to(device)
        pipeline = PipelineParallel(model, rank, world_size, device, chunks)
        print(f"Rank {rank} initialized")
    else:
        raise RuntimeError(f"Invalid rank {rank}")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    num_epochs = 3
    start_time = time.time()

    for epoch in range(num_epochs):
        if rank == 1:
            print(f"Epoch {epoch+1}/{num_epochs}")
        train_epoch(rank, world_size, pipeline, train_loader,
                    optimizer, criterion, device)

    if rank == 1:
        end = time.time()
        print(f"Parallel Pipeline Training time : {
              end - start_time:.2f} seconds")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
