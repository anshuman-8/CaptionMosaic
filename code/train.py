import torch
from tqdm.auto import tqdm
from torchtext.data.metrics import bleu_score
import matplotlib.pyplot as plt


def train(model, dataset, train_loader, val_loader, epochs, criterion, optimizer, scheduler, device):
    train_loss = []
    validation_loss =[]
    all_bleus = []
    best_bleu = 0.0

    for epoch in range(epochs):
        total_train = 0
        running_train_loss = 0.0
        print(f'Epoch: {epoch +1}')
        model.train()

        for images, captions in tqdm(train_loader, desc="Train ", position=0, leave=True):
            images = images.to(device)
            captions = captions.to(device)
           
            optimizer.zero_grad()
            output = model(images, captions[:-1]) 
            loss = criterion(output.reshape(-1, output.shape[2]), captions.reshape(-1))
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()
            total_train +=  1

        scheduler.step()

        model.eval()
        total_val = 0
        running_val_loss = 0.0
        epoch_bleu = []
        with torch.no_grad():
            for images, captions in tqdm(val_loader, desc="Validate",position=0, leave=True):
                images = images.to(device)
                captions = captions.to(device)

                v_output = model(images, captions[:-1])
                v_loss = criterion(v_output.reshape(-1, v_output.shape[2]), captions.reshape(-1))
                ref = captions.permute(1,0)
                output_prob = v_output.permute(1,0,2)
                cand = torch.argmax(output_prob, dim=2)
                batch_bleu = []
                for i in range(cand.size(0)):  
                    candidate_sentence = [dataset.vocab.itos[word.item()] for word in cand[i]]
                    reference_sentence = [dataset.vocab.itos[word.item()] for word in ref[i]]
                    bleu = bleu_score([candidate_sentence], [[reference_sentence]], max_n=4, weights=[0.10, 0.25, 0.30, 0.35])
                    batch_bleu.append(bleu)

                running_val_loss += v_loss.item()
                total_val += 1
                epoch_bleu.append(sum(batch_bleu) / len(batch_bleu))

        if epoch_bleu[-1] > best_bleu:
            best_bleu = epoch_bleu[-1]
            torch.save(model.state_dict(), '/home/jovyan/work/workspace/CaptionMosaic/models/model30k.pth')
        train_loss.append((running_train_loss/total_train))
        validation_loss.append((running_val_loss/total_val))
        all_bleus.append((sum(epoch_bleu) / len(epoch_bleu))* 100)
        print(f'Train Loss: {(running_train_loss/total_train):.4f}, Validation Loss: {(running_val_loss/total_val):.4f}, BLEU Score: {(sum(epoch_bleu) / len(epoch_bleu))* 100:.3f}\n')

    return train_loss, validation_loss, all_bleus


def plot_losses_and_bleus(train_loss, validation_loss, all_bleus):
    # Create an array of epoch numbers based on the length of the lists
    epochs = list(range(1, len(train_loss) + 1))

    # Create subplots for train loss, validation loss, and BLEU scores
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))

    # Plot training loss
    ax1.plot(epochs, train_loss, label='Training Loss', marker='o', linestyle='-')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss')
    ax1.set_title('Training Loss vs. Epoch')
    ax1.legend()

    # Plot validation loss
    ax2.plot(epochs, validation_loss, label='Validation Loss', marker='o', linestyle='-')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Validation Loss')
    ax2.set_title('Validation Loss vs. Epoch')
    ax2.legend()

    # Plot BLEU scores
    ax3.plot(epochs, all_bleus, label='BLEU Score', marker='o', linestyle='-')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('BLEU Score')
    ax3.set_title('BLEU Score vs. Epoch')
    ax3.legend()

    plt.tight_layout()
    plt.show()
