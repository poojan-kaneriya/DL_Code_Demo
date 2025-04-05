import torch
from utils.helpers import get_device


def evaluate(model, testloader):
    correct, total = 0, 0
    all_preds = []
    all_labels = []
    model.eval()
    device = get_device()
    model.to(device)

    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    accuracy = 100 * correct / total
    print(f"Accuracy: {accuracy:.2f}%")
    return all_labels, all_preds, accuracy