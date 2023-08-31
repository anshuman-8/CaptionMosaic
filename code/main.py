import torch
import torchvision.transforms as transforms
from data_loader import get_loader


# Hyperparameters
TRAIN_RATIO = 0.7
TEST_RATIO = 0.15
VAL_RATIO = 0.15
SPLIT = [0.7, 0.15, 0.15]

BATCH_SIZE = 32
WORKERS = 4
LEARNING_RATE=0.01
EMBED_SIZE = 256
HIDDEN_SIZE = 256
NUM_LAYERS = 1
EPOCHS = 3


transform = transforms.Compose([
        transforms.Resize((356, 356)),
        transforms.RandomCrop((224, 224)), 
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

train_loader, val_loader, test_loader, dataset = get_loader(
        "../Data/Images/", "../Data/captions.txt", transform=transform, num_workers=WORKERS, 
        batch_size=BATCH_SIZE, num_workers=WORKERS)
vocab_size = len(dataset)
