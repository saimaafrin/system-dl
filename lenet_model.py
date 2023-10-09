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

        ''' # Xavier/Glorot initialization
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)'''

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
        #print("Output after fc1:", x1)
        x2 = F.relu(self.fc2(x1))
        #print("Output after fc2:", x2)
        x3 = self.fc3(x2)
        #print("Output after fc3:", x3)
        return x3

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


# Set Device
device = torch.device('cuda')

# Hyperparameters
learning_rate = 0.0001
batch_size = 128
epochs = 200
#torch.set_printoptions(edgeitems=28)

# Load train Data
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
# Load test data
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


#examples = enumerate(train_loader)
#batch_idx, (example_data, example_targets) = next(examples)

#print(example_data.shape)
#print(example_data[0])

# Initialize Model
model = LeNet300(device)
model.to(device)

optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# Training Loop
for epoch in range(epochs):
    model.train()  # Set the model to training mode
    correct_train = 0
    total_train = 0
    for idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        # Forward pass
        output = model.forward(data)
        loss = F.cross_entropy(output, target)
        
        # Compute training accuracy
        _, predicted = torch.max(output.data, 1)
        total_train += target.size(0)
        correct_train += (predicted == target).sum().item()
        
        # Backward pass and optimization
        optimizer.zero_grad()  # Zero the gradients
        loss.backward()  # Computes the gradient of loss w.r.t parameters
        optimizer.step()  # Updates the parameters
        
        if idx % 100 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Step [{idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
    
    train_accuracy = 100 * correct_train / total_train
    print(f'Epoch [{epoch+1}/{epochs}], Train Accuracy: {train_accuracy:.2f}%')
    
    # Evaluate the model on the test data after each epoch
    test_accuracy = evaluate(model, test_loader, device)
    print(f'Epoch [{epoch+1}/{epochs}], Test Accuracy: {test_accuracy:.2f}%')

