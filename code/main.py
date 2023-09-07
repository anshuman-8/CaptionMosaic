import torch
import torchvision.transforms as transforms
from data_loader import get_loader
from model import CNNtoRNN, EncoderCNN, DecoderRNN
from train import train, plot_losses_and_bleus
import torch.optim as optim
import torch.nn as nn


# Hyperparameters
TRAIN_RATIO = 0.85
TEST_RATIO = 0.05
VAL_RATIO = 0.1

BATCH_SIZE = 32
WORKERS = 4
LEARNING_RATE=0.001
EMBED_SIZE = 324
HIDDEN_SIZE = 640
NUM_LAYERS = 3
EPOCHS = 70

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def main():
        transform = transforms.Compose([
        transforms.Resize((356, 356)),
        transforms.RandomCrop((224, 224)), 
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

        train_loader, val_loader, test_loader, dataset = get_loader(
                "/home/jovyan/work/workspace/CaptionMosaic/Data/Images/", "/home/jovyan/work/workspace/CaptionMosaic/Data/captions30k.txt", transform=transform, num_workers=WORKERS, 
                batch_size=BATCH_SIZE, split={"train":TRAIN_RATIO, "val":VAL_RATIO, "test":TEST_RATIO})
        vocab_size = len(dataset.vocab)

        model = CNNtoRNN(embed_size=EMBED_SIZE, hidden_size=HIDDEN_SIZE, vocab_size=vocab_size, num_layers=NUM_LAYERS, device=device).to(device)

        criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi['<PAD>']).to(device)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=0.000001)


        train_loss, validation_loss, all_bleus = train(model, dataset, train_loader, val_loader, EPOCHS, criterion, optimizer, scheduler, device )
        print("Training Complete !!\n")
        plot_losses_and_bleus(train_loss, validation_loss, all_bleus)


if __name__ == "__main__":
    main()
