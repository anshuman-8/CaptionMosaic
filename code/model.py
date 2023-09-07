import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size, train_CNN=False):
        super(EncoderCNN, self).__init__()
        self.train_CNN = train_CNN

        resnet101 = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V2)
        modules = list(resnet101.children())[:-2]
        self.resnet = nn.Sequential(*modules)

        self.adaptive_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(2048, embed_size),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )
        self.fine_tune(fine_tune=False)

    def forward(self, images):
        out = self.resnet(images)
        out = self.adaptive_pool(out)
        out = out.view(out.shape[0], -1)
        out = self.fc(out)
        return out
    
    def fine_tune(self, fine_tune=True):

        for p in self.resnet.parameters():
            p.requires_grad = False
        for c in list(self.resnet.children())[7:]:
            for p in c.parameters():
                p.requires_grad = fine_tune

    
    def fine_tune(self, fine_tune=True):

        for p in self.resnet.parameters():
            p.requires_grad = False
        for c in list(self.resnet.children())[6:]:
            for p in c.parameters():
                p.requires_grad = fine_tune



class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, max_seq_length=40):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=False)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.3)

    def forward(self, features, captions):
        embeddings = self.dropout1(self.embed(captions))
        # embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(0), embeddings), dim=0)
        hiddens, _ = self.lstm(embeddings)
        hiddens = self.dropout2(hiddens)
        outputs = self.linear(hiddens)
        return outputs



class CNNtoRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, device):
        super(CNNtoRNN, self).__init__()
        self.encoderCNN = EncoderCNN(embed_size).to(device)
        self.decoderRNN = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers).to(device)

    def forward(self, images, captions):
        features = self.encoderCNN(images)
        outputs = self.decoderRNN(features, captions)
        return outputs
    
    def caption_image(self, image, vocabulary, max_length=25):
        result_caption = []
        with torch.no_grad():
            x = self.encoderCNN(image).unsqueeze(0) # so that we have a dimention for batch
            states = None
            for _ in range(max_length):
                hiddens, states = self.decoderRNN.lstm(x,states)
                output = self.decoderRNN.linear(hiddens.squeeze(0))
                predicted = output.argmax(1) # take the word with the highest probability

                result_caption.append(predicted.item())
                x = self.decoderRNN.embed(predicted).unsqueeze(0)

                if vocabulary.itos[predicted.item()] == "<EOS>":
                    break
                
        return [vocabulary.itos[idx] for idx in result_caption]
    
    def beam_search(self, image, vocabulary, max_length = 25, beam_width=5):
        image = image.to(device)
        image = image.unsqueeze(0)

        beams = [{"sequence":[vocabulary["<START>"]], "score": 0.0}]

        completed_beams = []

        for _ in range(max_length):
            new_beams = []
            for beam in beams:
                current_sequence = beam["sequence"]
                current_score = beam["score"]

                input_sequence = torch.LongTensor(current_sequence).unsqueeze(1).to(device)

                with torch.no_grad():
                    output = self(image, input_sequence)
                    last_word_probs = output[0, -1, :]

                # Get the top-k next words
                top_k_probs, top_k_indices = torch.topk(last_word_probs, self.beam_width)

                # Expand the beams
                for i in range(self.beam_width):
                    word_index = top_k_indices[i].item()
                    word_prob = top_k_probs[i].item()

                    new_sequence = current_sequence + [word_index]
                    new_score = current_score - torch.log(word_prob)

                    new_beams.append({"sequence": new_sequence, "score": new_score})

            # Sort beams by score and select top-k
            new_beams.sort(key=lambda x: x["score"])
            beams = new_beams[:self.beam_width]

            # Check for completed sequences
            for beam in beams:
                if beam["sequence"][-1] == vocabulary["<END>"]:
                    completed_beams.append(beam)
                    beams.remove(beam)

            if len(beams) == 0:
                break

        # Sort completed beams and return the best one
        completed_beams.sort(key=lambda x: x["score"])
        best_beam = completed_beams[0]
        best_sequence = best_beam["sequence"]

        # Convert the sequence of word indices to words
        caption = [vocabulary.itos[word_index] for word_index in best_sequence[1:-1]]  # Exclude <START> and <END>
        return " ".join(caption)