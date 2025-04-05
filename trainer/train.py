import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from utils.helpers import get_device


def train(model, trainloader, config):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    device = get_device()
    model.to(device)
    model.train()
    train_losses = []
    train_accuracies = []

    for epoch in range(config["epochs"]):
        running_loss = 0.0
        correct, total = 0, 0

        for images, labels in tqdm(trainloader, desc=f"Epoch {epoch+1}/{config['epochs']}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        avg_loss = running_loss / len(trainloader)
        accuracy = 100 * correct / total
        train_losses.append(avg_loss)
        train_accuracies.append(accuracy)
        print(f"Epoch [{epoch+1}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

    return train_losses, train_accuracies

