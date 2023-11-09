import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Step 1: Load and preprocess the dataset
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize the image to 128x128 pixels
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Assuming the dataset is structured in folders with the label as the folder name
train_dataset = ImageFolder(root='Agricultural-crops', transform=transform)
test_dataset = ImageFolder(root='Agricultural-crops', transform=transform)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# Step 2: Implement a CNN with proper architecture
class SimpleCNN(nn.Module):
    def __init__(self, activation_func='relu'):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)  # Batch normalization after the first convolution
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)  # Batch normalization after the second convolution
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Max pooling layer
        self.fc1 = nn.Linear(64 * 32 * 32, 512)
        self.fc2 = nn.Linear(512, len(train_dataset.classes))
        self.dropout = nn.Dropout(0.5)
        self.activation_func = activation_func.lower()

    def forward(self, x):
        # Apply first convolution, batch normalization, activation, and pooling
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.pool(x)

        # Apply second convolution, batch normalization, activation, and pooling
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation(x)
        x = self.pool(x)

        # Flatten the tensor for the fully connected layers
        x = x.view(-1, self.num_flat_features(x))

        # Apply first fully connected layer with activation and dropout
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)

        # Apply second fully connected layer to produce final output
        x = self.fc2(x)
        return x

    def activation(self, x):
        if self.activation_func == 'sigmoid':
            return torch.sigmoid(x)
        elif self.activation_func == 'relu':
            return F.relu(x)
        else:
            raise ValueError("Unsupported activation function")

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

# Initialize weights using two different methods
def initialize_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0, std=0.01)
        nn.init.constant_(m.bias, 0)


# Create the model instance
model = SimpleCNN(activation_func='relu')  # You can change to 'sigmoid' here
model.apply(initialize_weights)

# Rest of the code for training, loss calculation, and visualization...

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Initialize lists to track per-epoch metrics
train_losses = []
train_accuracies = []
test_accuracies = []

# Function to calculate accuracy
def calculate_accuracy(loader, model):
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

# Train the model
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        running_loss += loss.item()

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Calculate average loss and accuracy
    avg_loss = running_loss / len(train_loader)
    train_losses.append(avg_loss)
    train_accuracy = calculate_accuracy(train_loader, model)
    train_accuracies.append(train_accuracy)

    # Print epoch statistics
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {train_accuracy:.2f}%')

# Test the model
model.eval()  # Set the model to evaluation mode
test_accuracy = calculate_accuracy(test_loader, model)
test_accuracies.append(test_accuracy)
print(f'Test Accuracy: {test_accuracy:.2f}%')

# Visualizing the training loss
plt.figure(figsize=(10, 5))
plt.title("Training Loss Over Epochs")
plt.plot(range(1, num_epochs + 1), train_losses, label="Train Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

# Visualizing the training accuracy
plt.figure(figsize=(10, 5))
plt.title("Training Accuracy Over Epochs")
plt.plot(range(1, num_epochs + 1), train_accuracies, label="Train Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# Visualizing the testing accuracy
plt.figure(figsize=(10, 5))
plt.title("Testing Accuracy")
plt.plot(range(1, num_epochs + 1), [test_accuracy] * num_epochs, label="Test Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
