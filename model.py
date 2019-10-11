import torch
import torch.nn as nn
import torchvision.models as models

#https://pytorch.org/2018/04/22/0_4_0-migration-guide.html
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet152(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        self.init_weights()

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        features = self.bn(features)
        return features
    
    def init_weights(self):
        self.embed.weight.data.normal_(0.0, 0.02)
        self.embed.bias.data.fill_(0)
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_dim, vocab_size, num_layers=1, max_seq_len=20):
        super(DecoderRNN, self).__init__()
    
        
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.hidden_dim = hidden_dim
        
        
        #n n.LSTM - Applies a multi-layer long short-term memory (LSTM) RNN to an input sequence.
        #LSTM Parameters:
        #input_size – The number of expected features in the input x
        # hidden_dim – The number of features in the hidden state h
        # num_layers – Number of recurrent layers. E.g., setting num_layers=2 would mean stacking two LSTMs together to             #form a stacked LSTM, with the second LSTM taking in outputs of the first LSTM and computing the final results.             #Default: 1
        #bias – If False, then the layer does not use bias weights b_ih and b_hh. Default: True
        #batch_first – If True, then the input and output tensors are provided as (batch, seq, feature)
        #dropout – If non-zero, introduces a Dropout layer on the outputs of each LSTM layer except the last layer, with           #dropout probability equal to dropout. Default: 0
        #bidirectional – If True, becomes a bidirectional LSTM. Default: False
        self.lstm = nn.LSTM(embed_size, hidden_dim, num_layers, batch_first=True)
        
        #nn.Linear Applies a linear transformation to the incoming data: y=Ax+b
        #nn.Linear Parameters:	
        #in_features – size of each input sample
        #out_features – size of each output sample
        #bias – If set to False, the layer will not learn an additive bias. Default: True
        self.linear = nn.Linear(hidden_dim, vocab_size)  
        self.max_seq_len = max_seq_len
        self.init_weights()

    def initHidden(self, batch_size):
        # reference:  https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros((1, batch_size, self.hidden_dim), device=device), 
                torch.zeros((1, batch_size, self.hidden_dim), device=device))
    
    def init_weights(self):
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.linear.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.fill_(0)

    
    def forward(self, features, encoded_captions):
        batch_size = features.shape[0]
        self.hidden = self.initHidden(batch_size) 
        
        #We won't decode at the <end> position
        encoded_captions = encoded_captions[:, :-1]    
                
        #Embeddings
        embeddings = self.embedding(encoded_captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1) 
        
        output, self.hidden = self.lstm(embeddings, self.hidden)
        predictions = self.linear(output)

        return predictions

    def sample(self, features):
        """Generate captions using greedy search."""
        
        sampled = []
        batch_size = features.shape[0]
        hidden = self.initHidden(batch_size) 
    
        while True:
            output, hidden = self.lstm(features, hidden)
            predictions = self.linear(output)
            predictions = predictions.squeeze(1)
            _, predicted = torch.max(predictions, dim=1) 
            sampled.append(predicted.cpu().numpy()[0].item()) 
            
            # Check for <end> word or max_seq_len
            if (predicted == 1) or len(sampled) > self.max_seq_len:    
                break
            
            features = self.embedding(predicted) 
            features = features.unsqueeze(1) 
            
        return sampled
