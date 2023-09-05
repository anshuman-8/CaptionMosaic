import torch
import torchvision.transforms as transforms
from data_loader import get_loader
from model import CNNtoRNN, EncoderCNN, DecoderRNN
from train import train
import torch.optim as optim
import torch.nn as nn


# Hyperparameters
TRAIN_RATIO = 0.7
TEST_RATIO = 0.15
VAL_RATIO = 0.15
SPLIT = [0.7, 0.15, 0.15]

BATCH_SIZE = 16
WORKERS = 4
LEARNING_RATE=0.01
EMBED_SIZE = 256
HIDDEN_SIZE = 512
NUM_LAYERS = 3
EPOCHS = 90

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def main():
        transform = transforms.Compose([
        transforms.Resize((356, 356)),
        transforms.RandomCrop((224, 224)), 
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

        train_loader, val_loader, test_loader, dataset = get_loader(
                "/home/jovyan/work/workspace/CaptionMosaic/Data/Images/", "/home/jovyan/work/workspace/CaptionMosaic/Data/captions.txt", transform=transform, num_workers=WORKERS, 
                batch_size=BATCH_SIZE)
        vocab_size = len(dataset.vocab)

        model = CNNtoRNN(embed_size=EMBED_SIZE, hidden_size=HIDDEN_SIZE, vocab_size=vocab_size, num_layers=NUM_LAYERS, device=device).to(device)

        criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi['<PAD>']).to(device)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=0.0001)


        train(model, dataset, train_loader, val_loader, EPOCHS, criterion, optimizer, scheduler, device)
        print("Training Complete !!")


if __name__ == "__main__":
    main()
