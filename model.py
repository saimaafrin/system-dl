import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from linear import LinearLayer

# Define the LeNet-300-100 Model using the custom LinearLayer
class LeNet300:
    def __init__(self, device):
        self.fc1 = LinearLayer(28*28, 300, device)
        self.fc2 = LinearLayer(300, 100, device)
        self.fc3 = LinearLayer(100, 10, device)
        self.device = device
    
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        x = F.relu(self.fc1.forward(x))
        x = F.relu(self.fc2.forward(x))
        x = self.fc3.forward(x)
        return x
    
    def backward(self, dY):
        dY = self.fc3.backward(dY)
        dY = F.relu_backward(dY, self.fc2.forward(self.fc1.forward(self.X.view(self.X.size(0), -1))))
        dY = self.fc1.backward(dY)
        return dY
    
    def update(self, lr):
        self.fc1.update(lr)
        self.fc2.update(lr)
        self.fc3.update(lr)

# Set Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
learning_rate = 0.001
batch_size = 64
epochs = 10

# Load Data
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# Initialize Model
model = LeNet300(device)

# Training Loop
for epoch in range(epochs):
    for idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        # Forward pass
        output = model.forward(data)
        loss = F.cross_entropy(output, target)
        
        # Backward pass and optimization
        dY = F.cross_entropy_backward(loss, target)
        model.backward(dY)
        model.update(learning_rate)
        
        if idx % 100 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Step [{idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
