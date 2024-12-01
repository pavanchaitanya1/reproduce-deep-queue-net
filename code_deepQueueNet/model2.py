import torch
import torch.nn as nn
import math
from attention import SelfAttention


class DeepQueueNet(nn.Module):
    def __init__(self, config, device):
        super(DeepQueueNet, self).__init__()
        self.config = config
        self.device = device
        
        # Positional Encoding
        self.pos_encoder = PositionalEncoding(12, config.dropout)
        
        # Transformer Encoder layers replacing LSTM1 and LSTM2
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=12,  # Input dimension
            nhead=config.transformer_params['n_heads'],
            dim_feedforward=config.transformer_params.get('dim_feedforward', 2048),
            dropout=config.dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.transformer_params.get('n_layers', 2)
        )
        
        # Linear layer to match dimensions
        self.input_proj = nn.Linear(12, config.lstm_params['cell_neurons'][1])
        
        if not self.config.use_transformer:
            self.fc1 = nn.Linear(12, self.config.lstm_params['cell_neurons'][1])
            self.fc2 = nn.Linear(self.config.mul_head_output_nodes, 
                               self.config.lstm_params['cell_neurons'][1])
        
        self.q_linear = nn.Linear(self.config.lstm_params['cell_neurons'][1], self.config.mul_head_output_nodes)
        
        # Multi-head self attention layers
        self.atts = nn.ModuleList([
            SelfAttention(self.config.lstm_params['cell_neurons'][1],
                         self.config.att,
                         self.config.att) for i in range(self.config.mul_head)
        ])
        
        self.encoder_o = nn.Linear(self.config.att * self.config.mul_head, 
                                 self.config.mul_head_output_nodes)

        # Transformer Decoder layers replacing LSTM3 and LSTM4
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.config.mul_head_output_nodes,
            nhead=config.transformer_params['n_heads'],
            dim_feedforward=config.transformer_params.get('dim_feedforward', 2048),
            dropout=config.dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=config.transformer_params.get('n_layers', 2)
        )
        
        self.decoder_o = nn.Linear(self.config.mul_head_output_nodes, self.config.n_outputs)
        self.stacked_o = []

    def forward(self, x):
        # Create attention mask if needed
        src_mask = None
        if self.config.mask:
            src_mask = generate_square_subsequent_mask(x.size(1)).to(self.device)
        
        if self.config.use_transformer:
            # Add positional encoding
            x = self.pos_encoder(x)
            # Pass through transformer encoder
            x = self.transformer_encoder(x, src_mask)
            x = self.input_proj(x)
        else:
            x = self.fc1(x)
        
        # Multi-head self attention
        self.X = torch.Tensor([]).to(self.device)
        for i, l in enumerate(self.atts):
            self.X = torch.cat((self.X, l(x)), dim=-1)
        
        self.X = self.encoder_o(self.X)
        self.X = self.X.reshape(-1, self.config.TIME_STEPS, self.config.mul_head_output_nodes)
        # print("X shape:", self.X.shape, x.shape)
        
        q = self.q_linear(x)
        # print("q shape:", q.shape)
        if self.config.use_transformer:
            # Create target mask for decoder
            # tgt_mask = generate_square_subsequent_mask(self.X.size(1)).to(self.device)
            # Pass through transformer decoder
            self.X = self.transformer_decoder(self.X, q)
        else:
            self.X = self.fc2(self.X)
        
        self.X = self.X.reshape(-1, self.config.mul_head_output_nodes)
        self.X = self.decoder_o(self.X)
        self.X = self.X.reshape(-1, self.config.TIME_STEPS, self.config.n_outputs)
        outputs = self.X[:, self.config.TIME_STEPS - 1, :]
        return outputs

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask