import torch
import tqdm
from torchtext.data.metrics import bleu_score

def train(model, dataset, train_loader, val_loader, epochs, criterion, optimizer, scheduler, device):
    train_loss = []
    epoch_bleus = []
    for epoch in range(epochs):
        total_train = 0
        running_train_loss = 0.0
        running_acc = 0.0
        print(f'Epoch: {epoch +1}')
        model.train()
        for images, captions, caplen in tqdm(train_loader, desc="Train\t"):
            images = images.to(device)
            captions = captions.to(device)
           
            output = model(images, captions[:-1]) 
            loss = criterion(output.reshape(-1, output.shape[2]), captions.reshape(-1))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            running_train_loss += loss.item()
            total_train +=  1

        scheduler.step()

        model.eval()
        total_val = 0
        running_val_loss = 0.0
        epoch_bleu = 0.0
        with torch.no_grad():
            for images, captions, _ in tqdm(val_loader, desc="Validate\t"):
                images = images.to(device)
                captions = captions.to(device)

                output = model(images, captions[:-1])
                loss = criterion(output.reshape(-1, output.shape[2]), captions.reshape(-1))
                ref = captions.permute(1,0)
                output_prob = output.permute(1,0,2)
                cand = torch.argmax(output_prob, dim=2)
                references_corpus = [[dataset.vocab.itos[word.item()] for word in sentence] for sentence in ref]
                candidate_corpus =  [[dataset.vocab.itos[word.item()] for word in sentence] for sentence in cand]
                # score = bleu_score(candidate_corpus, references_corpus)
                for candidate_sen, reference_sen in zip(candidate_corpus, references_corpus):
                    epoch_bleu += bleu_score(candidate_sen, reference_sen)
                
                running_val_loss += loss.item()
                total_val += 1
            
        epoch_bleus.append(epoch_bleu/total_val)

        if test_acc[-1] > best_acc:
            best_acc = test_acc[-1]
            torch.save(model.state_dict(), './models/model.pth') 

        print(f'Train Loss: {running_train_loss/total_train}, Validation Loss: {running_val_loss/total_val}, BLEU Score: {epoch_bleus[-1]}')