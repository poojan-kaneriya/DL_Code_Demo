import yaml
from data.dataloader import get_cifar10_loaders
from models.simple_cnn import SimpleCNN
from trainer.train import train
from trainer.evaluate import evaluate
from utils.helpers import set_seed

if __name__ == "__main__":
    with open("config/config.yaml") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["training"]["seed"])
    trainloader, testloader = get_cifar10_loaders(cfg["training"]["batch_size"])

    model = SimpleCNN()
    train(model, trainloader, cfg["training"])
    evaluate(model, testloader)