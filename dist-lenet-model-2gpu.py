#!/usr/bin/env python
import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from math import ceil
from random import Random
from torch.multiprocessing import Process
from torchvision import datasets, transforms


class Partition(object):
    """ Dataset-like object, but only access a subset of it. """
    """ See: https://pytorch.org/tutorials/intermediate/dist_tuto.html#distributed-training """

    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]


class DataPartitioner(object):
    """ Partitions a dataset into different chuncks. """

    def __init__(self, data, sizes=[0.7, 0.2, 0.1], seed=1234):
        self.data = data
        self.partitions = []
        rng = Random()
        rng.seed(seed)
        data_len = len(data)
        indexes = [x for x in range(0, data_len)]
        rng.shuffle(indexes)

        for frac in sizes:
            part_len = int(frac * data_len)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]

    def use(self, partition):
        return Partition(self.data, self.partitions[partition])


class LeNet300(nn.Module):
    """ LeNet-300-100 MLP Network. """

    def __init__(self, device):
        super(LeNet300, self).__init__()
        self.fc1 = nn.Linear(28*28, 300, device)
        self.fc2 = nn.Linear(300, 100, device)
        self.fc3 = nn.Linear(100, 10, device)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        x1 = F.relu(self.fc1(x))
        x2 = F.relu(self.fc2(x1))
        x3 = self.fc3(x2)
        return x3


def partition_dataset(batch_size):
    """ Partitioning MNIST training set """
    dataset = datasets.MNIST(
        './data',
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307, ), (0.3081, ))
        ]))
    size = dist.get_world_size()
    proc_bsz = batch_size // size
    partition_sizes = [1.0 / size for _ in range(size)]
    # print("word_size", size, "partition_sizes:", partition_sizes, "proc_bsz:", proc_bsz)
    partition = DataPartitioner(dataset, partition_sizes)
    partition = partition.use(dist.get_rank())
    train_set = torch.utils.data.DataLoader(
        partition, batch_size=proc_bsz, shuffle=True)
    return train_set, proc_bsz


def get_test_loader(batch_size):
    """ Return the test loader of MNIST """
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    test_dataset = datasets.MNIST(
        root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=False)
    return test_loader


def average_gradients(model):
    """ Gradient averaging. """
    size = float(dist.get_world_size())
    for param in model.parameters():
        if type(param) is torch.Tensor:
            dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM, group=0)
            param.grad.data /= size


def evaluate(model, test_loader, device):
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():  # Disable gradient computation during evaluation
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    accuracy = 100 * correct / total
    return accuracy


def cleanup():
    dist.destroy_process_group()


def run(rank, size, device_type, learning_rate, batch_size, epochs):
    """ Distributed Synchronous SGD """
    torch.manual_seed(1234)
    train_loader, _ = partition_dataset(batch_size)
    test_loader = get_test_loader(batch_size)
    device = torch.device(f"{device_type}:{rank}")
    print("device: ", device)
    model = LeNet300(device)
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    # Training Loop
    for epoch in range(epochs):
        model.train()  # Set the model to training mode
        epoch_loss = 0.0
        correct_train, total_train = 0, 0
        for idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            # Forward pass
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            epoch_loss += loss

            # Compute training accuracy
            _, predicted = torch.max(output.data, 1)
            total_train += target.size(0)
            correct_train += (predicted == target).sum().item()

            # Backward pass
            loss.backward()  # Zero the gradients
            average_gradients(model)  # Sync the gradients by averaging them
            optimizer.step()  # Update the parameters
            # if idx % 100 == 0:
            # print(f'Epoch [{epoch+1}/{epochs}], Step [{idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
        train_accuracy = 100 * correct_train / total_train
        test_accuracy = evaluate(model, test_loader, device)
        print(
            f'Rank {dist.get_rank()}, Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss.item() / len(train_loader):.3f}, Train Accuracy: {train_accuracy:.2f}%, Test Accuracy: {test_accuracy:.2f}%')
    cleanup()


def init_processes(rank, size, device_type, learning_rate, batch_size, epochs, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size, device_type, learning_rate, batch_size, epochs)


if __name__ == "__main__":
    # Declare Device Type
    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("device_type: ", device_type)
    # Hyperparameters
    learning_rate = 0.01
    batch_size = 128
    epochs = 10
    # 2 processes/nodes
    size = 2
    processes = []
    # https://pytorch.org/docs/stable/notes/multiprocessing.html#cuda-in-multiprocessing
    torch.multiprocessing.set_start_method('spawn')
    for rank in range(size):
        p = Process(target=init_processes, args=(
            rank, size, device_type, learning_rate, batch_size, epochs, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
