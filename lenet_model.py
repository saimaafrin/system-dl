import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from custom_linear import CustomLinearLayer
import torch.optim as optim


# Define the LeNet-300-100 Model using the custom LinearLayer

class LeNet300(nn.Module):
    def __init__(self, device):
        super(LeNet300, self).__init__()
        self.fc1 = CustomLinearLayer(28*28, 300, device)
        self.fc2 = CustomLinearLayer(300, 100, device)
        self.fc3 = CustomLinearLayer(100, 10, device)

         # Xavier/Glorot initialization
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)

    '''def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        
        x1 = F.relu(self.fc1(x))
        if torch.isnan(x1).any():
            print("NaN values found after fc1")
            x1 = torch.where(torch.isnan(x1), torch.zeros_like(x1), x1)
        
        x2 = F.relu(self.fc2(x1))
        if torch.isnan(x2).any():
            print("NaN values found after fc2")
            x2 = torch.where(torch.isnan(x2), torch.zeros_like(x2), x2)
        
        x3 = self.fc3(x2)
        if torch.isnan(x3).any():
            print("NaN values found after fc3")
            x3 = torch.where(torch.isnan(x3), torch.zeros_like(x3), x3)
        
        return x3'''

        
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        x1 = F.relu(self.fc1(x))
        print("Output after fc1:", x1)
        x2 = F.relu(self.fc2(x1))
        print("Output after fc2:", x2)
        x3 = self.fc3(x2)
        print("Output after fc3:", x3)
        return x3

# Set Device
device = torch.device('cuda')

# Hyperparameters
learning_rate = 0.0001
batch_size = 64
epochs = 10

# Load Data
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# Initialize Model
model = LeNet300(device)
#model.to(device)

optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# Training Loop
for epoch in range(epochs):
    for idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        # Forward pass
        output = model.forward(data)
        loss = F.cross_entropy(output, target)
        
        # Backward pass and optimization
        loss.backward()  # Computes the gradient of loss w.r.t parameters
        optimizer.step()  # Updates the parameters
        #torch.cuda.empty_cache()
        if idx % 100 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Step [{idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
