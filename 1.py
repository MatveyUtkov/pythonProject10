# Import necessary libraries
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Data augmentation and normalization for training
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load the datasets
train_dataset = ImageFolder(root='Agricultural-crops', transform=transform)
test_dataset = ImageFolder(root='Agricultural-crops', transform=transform)

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define the CNN architecture with a parameter to specify the activation function
class SimpleCNN(nn.Module):
    def __init__(self, num_classes, activation_func='relu'):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 32 * 32, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)
        self.activation_func = activation_func.lower()

    def forward(self, x):
        x = self.pool(self.bn1(self.conv1(x)))
        x = self.activation(x)
        x = self.pool(self.bn2(self.conv2(x)))
        x = self.activation(x)
        x = x.view(-1, 64 * 32 * 32)
        x = self.dropout(self.activation(self.fc1(x)))
        x = self.fc2(x)
        return x

    def activation(self, x):
        return torch.sigmoid(x) if self.activation_func == 'sigmoid' else F.relu(x)

# Weight initialization functions
def initialize_weights_xavier(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
def initialize_weights_small_random(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0.0, std=0.01)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

# Create models with ReLU and Sigmoid activations
num_classes = len(train_dataset.classes)
model_relu = SimpleCNN(num_classes=num_classes, activation_func='relu')
model_sigmoid = SimpleCNN(num_classes=num_classes, activation_func='sigmoid')

# Initialize weights with Xavier method
model_relu.apply(initialize_weights_xavier)
model_sigmoid.apply(initialize_weights_xavier)
# Initialize weights with small random numbers
model_relu.apply(initialize_weights_small_random)
model_sigmoid.apply(initialize_weights_small_random)

# Define loss function and optimizers
criterion = nn.CrossEntropyLoss()
optimizer_relu = optim.Adam(model_relu.parameters(), lr=0.001)
optimizer_sigmoid = optim.Adam(model_sigmoid.parameters(), lr=0.001)

# Define learning rate schedulers
scheduler_relu = optim.lr_scheduler.StepLR(optimizer_relu, step_size=7, gamma=0.1)
scheduler_sigmoid = optim.lr_scheduler.StepLR(optimizer_sigmoid, step_size=7, gamma=0.1)

# calculate accuracy
def calculate_accuracy(loader, model):
    model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

# train function
def train_model(model, train_loader, optimizer, criterion, scheduler, num_epochs=10):
    model.train()  # Set model to training mode
    loss_history = []
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        scheduler.step()  # Adjust the learning rate
        epoch_loss = running_loss / len(train_loader)
        loss_history.append(epoch_loss)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')
    return loss_history

# train and evaluate
def train_and_evaluate(model, optimizer, scheduler, activation_func):
    print(f"Training and evaluating the model with {activation_func} activation...")
    loss_history = train_model(model, train_loader, optimizer, criterion, scheduler, num_epochs=10)
    accuracy = calculate_accuracy(test_loader, model)
    print(f"{activation_func} Model Test Accuracy: {accuracy:.2f}%")
    return loss_history, accuracy

# Train and evaluate the ReLU model
relu_loss_history, relu_accuracy = train_and_evaluate(model_relu, optimizer_relu, scheduler_relu, 'ReLU')

# Train and evaluate the Sigmoid model
sigmoid_loss_history, sigmoid_accuracy = train_and_evaluate(model_sigmoid, optimizer_sigmoid, scheduler_sigmoid, 'Sigmoid')

# Visualization
plt.figure(figsize=(12, 5))

# Plot training loss history
plt.subplot(1, 2, 1)
plt.plot(relu_loss_history, label='ReLU Loss')
plt.plot(sigmoid_loss_history, label='Sigmoid Loss')
plt.title('Training Loss History')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot test accuracy
plt.subplot(1, 2, 2)
plt.bar(['ReLU', 'Sigmoid'], [relu_accuracy, sigmoid_accuracy], color=['blue', 'orange'])
plt.title('Test Accuracy')
plt.xlabel('Activation Function')
plt.ylabel('Accuracy (%)')

plt.tight_layout()
plt.show()