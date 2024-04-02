import torch
from torch import nn
import torch.nn.functional as F
import random
import numpy as np
import argparse 
import os
import pdb

parser = argparse.ArgumentParser(description='pretrain seq2seq by smiles data')    
parser.add_argument('-n', '--gpu', help='GPU number',type=str,required=True)   
parser.add_argument('-f', '--file', help='save file',type=str,required=True)   

args = parser.parse_args()  
filesave = args.file
GPU_NUM =  args.gpu# Number of GPU


os.environ["CUDA_VISIBLE_DEVICES"]=GPU_NUM

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print('Device:', device)
print('Current cuda device:', torch.cuda.current_device())
print('Count of using GPUs:', torch.cuda.device_count())

print("Using Device:", device)
# pdb.set_trace()
torch.manual_seed(123)
torch.cuda.manual_seed_all(123)
print("CPU Random Seed :", torch.initial_seed())
print("CUDA Random Seed :", torch.cuda.initial_seed())

modelArgs = {}
modelArgs['INPUT_DIM'] = 68
modelArgs['ENC_EMB_DIM'] = 128
modelArgs['HID_DIM'] = 128
modelArgs['OUTPUT_DIM'] = 68
modelArgs['DEC_EMB_DIM'] = 128
modelArgs['ENC_DROPOUT'] = 0.2
modelArgs['DEC_DROPOUT'] = 0.2
modelArgs['epochs'] = 30000
modelArgs['filesave'] = filesave
modelArgs['N_LAYERS'] = 2
modelArgs['batch_size'] = 256


# input_dim
# emb_dim = 68
# hid_dim = 128 


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout):
        super().__init__()
        
        self.embedding = nn.Embedding(input_dim, emb_dim)
        
        # bidirectional=True로 설정하여 bi-rnn을 구현합니다.
        self.rnn = nn.GRU(emb_dim, enc_hid_dim, bidirectional = True)
        
        # 양방향 rnn의 출력값을 concat 한 후에 fc layer에 전달합니다.
        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        
        #src = [src len, batch size]
        
        # 입력 x를 임베딩
        embedded = self.dropout(self.embedding(src.to(device)))
        
        #embedded = [src len, batch size, emb dim]
        
        # rnn의 출력값
        outputs, hidden = self.rnn(embedded)
        # pdb.set_trace()
        #outputs = [src len, batch size, hid dim * num directions]
        #hidden = [n layers * num directions, batch size, hid dim]
        
        #hidden is stacked [forward_1, backward_1, forward_2, backward_2, ...]
        #outputs are always from the last layer
        
        #hidden [-2, :, : ] is the last of the forwards RNN 
        #hidden [-1, :, : ] is the last of the backwards RNN
        
        #initial decoder hidden is final hidden state of the forwards and backwards 
        #  encoder RNNs fed through a linear layer
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)))
        
        #outputs = [src len, batch size, enc hid dim * 2]
        #hidden = [batch size, dec hid dim]
        
        return outputs, hidden


class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        
        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias = False)
        
    def forward(self, hidden, encoder_outputs):
        
        #hidden = [batch size, dec hid dim]
        #encoder_outputs = [src len, batch size, enc hid dim * 2]
        
        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]
        
        #repeat decoder hidden state src_len times
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        
        #hidden = [batch size, src len, dec hid dim]
        #encoder_outputs = [batch size, src len, enc hid dim * 2]
        
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim = 2))) 
        
        #energy = [batch size, src len, dec hid dim]

        attention = self.v(energy).squeeze(2)
        
        #attention= [batch size, src len]
        
        return F.softmax(attention, dim=1)


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention):
        super().__init__()

        self.output_dim = output_dim
        self.attention = attention
        
        self.embedding = nn.Embedding(output_dim, emb_dim)
        
        # embedding과 weighted vector가 concat 된 후, 이전 hidden staet와 함께 입력
        self.rnn = nn.GRU((enc_hid_dim * 2) + emb_dim, dec_hid_dim)
        
        # 입력값 d(y_t), w_t, s_t
        self.fc_out = nn.Linear((enc_hid_dim * 2) + dec_hid_dim + emb_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, encoder_outputs):
             
        #input = [batch size]
        #hidden = [batch size, dec hid dim]
        #encoder_outputs = [src len, batch size, enc hid dim * 2]
        
        input = input.unsqueeze(0)
        
        #input = [1, batch size]
        
        embedded = self.dropout(self.embedding(input.to(device)))
        
        #embedded = [1, batch size, emb dim]
        
        a = self.attention(hidden, encoder_outputs)
                
        #a = [batch size, src len]
        
        a = a.unsqueeze(1)
        
        #a = [batch size, 1, src len]
        
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        
        #encoder_outputs = [batch size, src len, enc hid dim * 2]
        
        weighted = torch.bmm(a, encoder_outputs)
        
        #weighted = [batch size, 1, enc hid dim * 2]
        
        weighted = weighted.permute(1, 0, 2)
        
        #weighted = [1, batch size, enc hid dim * 2]
        
        rnn_input = torch.cat((embedded, weighted), dim = 2)
        
        #rnn_input = [1, batch size, (enc hid dim * 2) + emb dim]
            
        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))
        
        #output = [seq len, batch size, dec hid dim * n directions]
        #hidden = [n layers * n directions, batch size, dec hid dim]
        
        #seq len, n layers and n directions will always be 1 in this decoder, therefore:
        #output = [1, batch size, dec hid dim]
        #hidden = [1, batch size, dec hid dim]
        #this also means that output == hidden
        assert (output == hidden).all()
        
        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)
        
        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim = 1))
        
        #prediction = [batch size, output dim]
        
        return prediction, hidden.squeeze(0)


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, src, trg, teacher_forcing_ratio = 0.5):
        
        #src = [src len, batch size]
        #trg = [trg len, batch size]
        #teacher_forcing_ratio is probability to use teacher forcing
        #e.g. if teacher_forcing_ratio is 0.75 we use teacher forcing 75% of the time
        
        batch_size = src.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        
        #tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size)
        
        #encoder_outputs is all hidden states of the input sequence, back and forwards
        #hidden is the final forward and backward hidden states, passed through a linear layer
        encoder_outputs, hidden = self.encoder(src)
                
        #first input to the decoder is the <sos> tokens
        input = trg[0,:]
        
        for t in range(1, trg_len):
            
            #insert input token embedding, previous hidden state and all encoder hidden states
            #receive output tensor (predictions) and new hidden state
            output, hidden = self.decoder(input, hidden, encoder_outputs)
            # pdb.set_trace()
            #place predictions in a tensor holding predictions for each token
            outputs[t] = output
            
            #decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio
            
            #get the highest predicted token from our predictions
            top1 = output.argmax(1) 
            
            #if teacher forcing, use actual next token as next input
            #if not, use predicted token
            input = trg[t] if teacher_force else top1
            # if t==170:
            #     pdb.set_trace()
        # outputs = outputs.argmax(2)
        # pdb.set_trace()
        return outputs



from torch.utils.data import Dataset, DataLoader

class SMILESDataset(Dataset):
    def __init__(self, src, trg):
        self.src = src
        self.trg = trg

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        return self.src[idx], self.trg[idx]

"""Output files"""
setting = modelArgs['filesave']
file_model_s2s = './results/model--seq2seq--' + setting+ '.pth'
file_model_enc = './results/model--enc--' + setting+ '.pth'

data_dir = "/NAS_Storage4/jaesuk/DeepAffinity2019/seq2seq_models/data/compound/"
model_dir = "/NAS_Storage4/jaesuk/DeepAffinity2019/seq2seq_models/data/compound/models/"
src_train = np.load(data_dir + "SMILE_from.txt.ids68.npy")
src_val = np.load(data_dir + "SMILE_from_dev.txt.ids68.npy")

trg_train = np.load(data_dir + "SMILE_from.txt.ids68.npy")
trg_val = np.load(data_dir + "SMILE_from_dev.txt.ids68.npy")

train_dataset = SMILESDataset(src_train, trg_train)
val_dataset = SMILESDataset(src_val, trg_val)

train_loader = DataLoader(train_dataset, batch_size=modelArgs['batch_size'], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=modelArgs['batch_size'], shuffle=False)


# Initialize the modules
enc = Encoder(modelArgs['INPUT_DIM'], modelArgs['ENC_EMB_DIM'], modelArgs['HID_DIM'], modelArgs['HID_DIM'], modelArgs['ENC_DROPOUT'])
attn = Attention(modelArgs['HID_DIM'], modelArgs['HID_DIM'])
dec = Decoder(modelArgs['OUTPUT_DIM'], modelArgs['DEC_EMB_DIM'], modelArgs['HID_DIM'], modelArgs['HID_DIM'], modelArgs['DEC_DROPOUT'], attn)

# Create the Seq2Seq model
model = Seq2Seq(enc, dec, device).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss(ignore_index=0)


best_loss = np.inf
for epoch in range(modelArgs['epochs']):
    model.train()  # 학습 모드
    for srg, trg in train_loader:
        srg_T = np.transpose(srg)
        trg_T = np.transpose(trg)
        
        optimizer.zero_grad()
        output = model(srg_T, trg_T)
        # pdb.set_trace()
        # output_T = np.transpose(output)
        output_dim = output.shape[-1]
            
        output = output[1:].view(-1, output_dim)
        # pdb.set_trace()
        trg_T = trg_T[1:].reshape(-1)
        # pdb.set_trace()
        loss = criterion(output, trg_T)
        loss.backward()
        optimizer.step()

    model.eval()  # 평가 모드
    with torch.no_grad():
        for srg, trg in val_loader:
            srg_T = np.transpose(srg)
            trg_T = np.transpose(trg)
            
            output = model(srg_T, trg_T)
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            trg_T = trg_T[1:].reshape(-1)
            
            val_loss = criterion(output, trg_T)
            # 추가적으로 accuracy 등의 지표를 계산할 수 있습니다.
    
    # 학습 상태 출력
    print(f"Epoch: {epoch}, Train Loss: {loss.item()}, Val Loss: {val_loss.item()}")
    
    if val_loss < best_loss:
        torch.save(model.encoder.state_dict(), file_model_enc)
        torch.save(model.state_dict(), file_model_s2s)
        best_loss = val_loss
        patience = 0
    else:
        patience+=1
    if patience == 20:
        print('stop')
        print(best_loss)
        break

