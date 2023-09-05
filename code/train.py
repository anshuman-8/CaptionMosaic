import torch
import tqdm
from torchtext.data.metrics import bleu_score

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

        for images, captions in tqdm(train_loader, desc="Train\t"):
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
            for images, captions in tqdm(val_loader, desc="Validate\t"):
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
                    bleu = bleu_score([candidate_sentence], [[reference_sentence]], max_n=4, weights=[0.15, 0.25, 0.35, 0.25])
                    batch_bleu.append(bleu)

                running_val_loss += v_loss.item()
                total_val += 1
                epoch_bleu.append(sum(batch_bleu) / len(batch_bleu))

        if epoch_bleu[-1] > best_bleu:
            best_bleu = epoch_bleu[-1]
            torch.save(model.state_dict(), '../models/model1.pth')
        train_loss.append((running_train_loss/total_train))
        validation_loss.append((running_val_loss/total_val))
        all_bleus.append((sum(epoch_bleu) / len(epoch_bleu))* 100)
        print(f'Train Loss: {(running_train_loss/total_train):.4f}, Validation Loss: {(running_val_loss/total_val):.4f}, BLEU Score: {(sum(epoch_bleu) / len(epoch_bleu))* 100:.2f}\n')

    return train_loss, validation_loss, all_bleus
