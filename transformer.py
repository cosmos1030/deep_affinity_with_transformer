import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

class PositionalEncoding(nn.Module):
    def __init__(self, position, d_model):
        super().__init__()
        self.pos_encoding = self.positional_encoding(position, d_model).to(device)
    
    def get_angles(self, position, i, d_model):
        angles = 1/ torch.pow(10000, (2 * (i//2)) / d_model)
        return position * angles
    
    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(
            position = torch.arange(position, device=device).unsqueeze(1),
            i = torch.arange(d_model, device=device).unsqueeze(0),
            d_model = d_model
        )
        sines = torch.sin(angle_rads[:, 0::2])
        cosines = torch.cos(angle_rads[:, 1::2])
        angle_rads = torch.zeros_like(angle_rads)
        angle_rads[:, 0::2] = sines
        angle_rads[:, 1::2] = cosines
        pos_encoding = angle_rads.unsqueeze(0)

        return pos_encoding
    def forward(self, inputs):
        return inputs + self.pos_encoding[:, :inputs.shape[1],:]
    

def scaled_dot_product_attention(query, key, value, mask=None):
    matmul_qk = torch.matmul(query, key.transpose(-2,-1))

    depth = query.size(-1)
    logits = matmul_qk / torch.sqrt(torch.tensor(depth, dtype=torch.float32))

    if mask is not None:
        logits += (mask * -1e9)
    
    attention_weights = F.softmax(logits, dim=-1)

    output = torch.matmul(attention_weights, value)
    return output, attention_weights

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        self.depth = d_model // self.num_heads

        self.query_dense = nn.Linear(d_model, d_model)
        self.key_dense = nn.Linear(d_model, d_model)
        self.value_dense = nn.Linear(d_model, d_model)

        self.dense = nn.Linear(d_model, d_model)

    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.permute(0,2,1,3)

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        query = self.query_dense(query)
        key = self.key_dense(key)
        value = self.value_dense(value)

        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        scaled_attention, _ = scaled_dot_product_attention(query, key, value, mask)
        scaled_attention = scaled_attention.permute(0,2,1,3)
        
        concat_attention = scaled_attention.contiguous().view(batch_size, -1, self.d_model)

        outputs = self.dense(concat_attention)
        return outputs

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dff, dropout):
        super().__init__()
    
        self.multi_head_attention = MultiHeadAttention(d_model, num_heads)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)

        self.dense1 = nn.Linear(d_model, dff)
        self.dense2 = nn.Linear(dff, d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)
    def forward(self, x, padding_mask):
        attn_output = self.multi_head_attention(x,x,x,padding_mask)
        attn_output = self.dropout1(attn_output)
        out1 = self.norm1(x+attn_output)
        
        ffn_output = F.relu(self.dense1(out1))
        ffn_output = self.dense2(ffn_output)
        ffn_output = self.dropout2(ffn_output)
        out2 = self.norm2(out1 + ffn_output)
        return out2


class Encoder(nn.Module):
    def __init__(self, vocab_size, seq_length, num_layers, dff, d_model, num_heads, dropout):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(seq_length, d_model)
        self.dropout = nn.Dropout(dropout)

        self.enc_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, dff, dropout) for _ in range(num_layers)])
        
    def forward(self, x, padding_mask):
        x = self.embedding(x)
        x *= torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        x = self.pos_encoding(x)
        x = self.dropout(x)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, padding_mask)
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dff, dropout):
        super().__init__()
        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)

        self.mha2 = MultiHeadAttention(d_model, num_heads)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)

        self.dense1 = nn.Linear(d_model, dff)
        self.dense2 = nn.Linear(dff, d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.norm3 = nn.LayerNorm(d_model, eps=1e-6)
    
    def forward(self, x, enc_output, look_ahead_mask, padding_mask):
        attn1 = self.mha1(x,x,x,look_ahead_mask)
        attn1 = self.norm1(attn1 + x)

        attn2 = self.mha2(attn1, enc_output, enc_output, padding_mask)
        attn2 = self.dropout1(attn2)
        attn2 = self.norm2(attn2 + attn1)

        ffn_output = F.relu(self.dense1(attn2))
        ffn_output = self.dense2(ffn_output)
        ffn_output = self.dropout2(ffn_output)
        out = self.norm3(ffn_output + attn2)
        return out

class Decoder(nn.Module):
    def __init__(self, vocab_size, seq_length, num_layers, dff, d_model, num_heads, dropout):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(seq_length, d_model)
        self.dropout = nn.Dropout(dropout)

        self.dec_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, dff, dropout)
            for _ in range(num_layers)
        ])
    def forward(self, x, enc_output, look_ahead_mask, padding_mask):

        x = self.embedding(x)
        x *= torch.sqrt(torch.tensor(self.d_model, dtype=torch.float))
        x += self.pos_encoding(x)
        x = self.dropout(x)

        for i in range(self.num_layers):
            x = self.dec_layers[i](x, enc_output, look_ahead_mask, padding_mask)
        return x

class Transformer(nn.Module):
    def __init__(self, protein_vocab_size, compound_vocab_size, protein_seq_length, compound_seq_length, num_layers, dff, protein_embedding_dim, compound_embedding_dim, num_heads, dropout):
        super().__init__()
        self.encoder = Encoder(protein_vocab_size, protein_seq_length, num_layers, dff, protein_embedding_dim, num_heads, dropout)
        self.layer1 = nn.Linear(protein_embedding_dim, compound_embedding_dim)  # 인코더와 디코더의 임베딩 차원 다른 문제 해결
        self.decoder = Decoder(compound_vocab_size, compound_seq_length, num_layers, dff, compound_embedding_dim, num_heads, dropout)
        self.final_layer = nn.Linear(compound_embedding_dim, 1)  # 최종적으로 IC50 값을 예측하기 위한 레이어

    def forward(self, inp, tar, enc_padding_mask, look_ahead_mask, dec_padding_mask):
        enc_output = self.encoder(inp, enc_padding_mask)
        inter_output1 = self.layer1(enc_output)
        dec_output = self.decoder(tar, inter_output1, look_ahead_mask, dec_padding_mask)

        # 디코더 출력에서 시퀀스 길이에 대해 평균을 내어 (batch_size, compound_embedding_dim) 모양을 얻습니다.
        pooled_output = torch.mean(dec_output, dim=1)

        # 평균 풀링된 출력을 최종 레이어에 통과시켜 IC50 값을 예측합니다.
        final_output = self.final_layer(pooled_output)  # (batch_size, 1) 모양의 출력
        return final_output


def create_padding_mask(x):
    # x와 0이 같은지 비교하여 마스크 생성 (x가 0이면 True, 아니면 False)
    mask = torch.eq(x, 0).float()
    # (batch_size, 1, 1, key의 문장 길이) 형태로 차원 변경
    return mask.unsqueeze(1).unsqueeze(2)
