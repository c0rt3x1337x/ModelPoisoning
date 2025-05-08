import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import random

# ----- Device Setup -----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----- CNN Model -----
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ----- Data Preparation -----
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Clean training data
clean_train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# ----- Function to Inject Label Noise (Poisoning) -----
def poison_labels(dataset, noise_ratio=0.1):
    indices = list(range(len(dataset)))
    num_noisy = int(noise_ratio * len(indices))
    noisy_indices = random.sample(indices, num_noisy)
    for idx in noisy_indices:
        true_label = dataset.targets[idx].item()
        new_label = random.choice([l for l in range(10) if l != true_label])
        dataset.targets[idx] = new_label
    return dataset

# ----- Poisoned Data -----
poisoned_dataset = poison_labels(datasets.MNIST(root='./data', train=True, download=True, transform=transform))
poisoned_train_loader = DataLoader(poisoned_dataset, batch_size=64, shuffle=True)

# ----- Training Function -----
def train_model(model, train_loader, num_epochs=5):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

# ----- Confusion Matrix -----
def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

# ----- Evaluation Function -----
def evaluate(model, data_loader, label="Model"):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print(f"\n=== Classification Report: {label} ===")
    print(classification_report(all_labels, all_preds, digits=4))
    plot_confusion_matrix(all_labels, all_preds, title=f"{label} Confusion Matrix")

# ----- Train & Evaluate Clean Model -----
print("\n=== Training Clean Model ===")
clean_model = SimpleCNN()
train_model(clean_model, clean_train_loader)
print("\n=== Evaluation: Clean Model ===")
evaluate(clean_model, test_loader, label="Clean Model")

# ----- Train & Evaluate Poisoned Model -----
print("\n=== Training Poisoned Model ===")
poisoned_model = SimpleCNN()
train_model(poisoned_model, poisoned_train_loader)
print("\n=== Evaluation: Poisoned Model ===")
evaluate(poisoned_model, test_loader, label="Poisoned Model")

# ----- Comparative Notes -----
print("""\n=== Comparative Analysis ===
- Clean model shows higher accuracy across all classes.
- Poisoned model suffers from misclassifications due to label noise.
- Confusion matrix highlights areas of confusion post-poisoning.
""")

