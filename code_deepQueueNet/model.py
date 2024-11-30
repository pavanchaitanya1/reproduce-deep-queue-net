import torch
import torch.nn as nn
from attention import SelfAttention


class DeepQueueNet(nn.Module):
    def __init__(self,config,device):
        super(DeepQueueNet, self).__init__()
        self.config = config
        #Encoder
        self.lstm1 = nn.LSTM(input_size=12, hidden_size=self.config.lstm_params['cell_neurons'][0], \
            num_layers=1, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=self.config.lstm_params['cell_neurons'][0], \
            hidden_size=self.config.lstm_params['cell_neurons'][1], num_layers=1, batch_first=True)
        if not self.config.lstm:
            self.fc1 = nn.Linear(12, self.config.lstm_params['cell_neurons'][1])
            self.fc2 = nn.Linear(self.config.mul_head_output_nodes, \
                self.config.lstm_params['cell_neurons'][1])
        self.q_linear = nn.Linear(self.config.lstm_params['cell_neurons'][1], self.config.att)
        # self.k_linear = nn.Linear(self.config.lstm_params['cell_neurons'][1],self.config.att)
        # self.v_linear = nn.Linear(self.config.lstm_params['cell_neurons'][1],self.config.att)

        # self.multihead_attn = nn.MultiheadAttention(self.config.att, \
        #     self.config.mul_head)
        self.atts = nn.ModuleList([SelfAttention(self.config.lstm_params['cell_neurons'][1],\
             self.config.att , self.config.att) for i in range(self.config.mul_head)])
        
        self.encoder_o = nn.Linear(self.config.att * self.config.mul_head, self.config.mul_head_output_nodes)

        #Decoder
        self.lstm3 = nn.LSTM(input_size=self.config.mul_head_output_nodes, hidden_size=self.config.lstm_params['cell_neurons'][0], \
            num_layers=1, batch_first=True)
        self.lstm4 = nn.LSTM(input_size=self.config.lstm_params['cell_neurons'][0], \
            hidden_size=self.config.lstm_params['cell_neurons'][1], num_layers=1, batch_first=True)

        self.decoder_o = nn.Linear(self.config.lstm_params['cell_neurons'][1],self.config.n_outputs)
        self.device = device
        self.stacked_o = []

    def forward(self, x):
        self.X = torch.Tensor([]).to(self.device)
        if self.config.lstm:
            x, (h,c) = self.lstm1(x)
            x, (h,c) = self.lstm2(x)
        else:
            x = self.fc1(x)
        for i, l in enumerate(self.atts):
            self.X = torch.cat((self.X,l(x)),dim=-1)
        
            #x = self.linears[i // 2](x) + l(x)
        self.X = self.encoder_o(self.X)
        
        
        self.X = self.X.reshape(-1, self.config.TIME_STEPS, self.config.mul_head_output_nodes)
        if self.config.lstm:
            self.X, (h,c) = self.lstm3(self.X)
            self.X, (h,c) = self.lstm4(self.X)
        else:
            self.X = self.fc2(self.X)
        self.X = self.X.reshape(-1,self.config.lstm_params['cell_neurons'][-1])
        self.X = self.decoder_o(self.X)
        self.X = self.X.reshape(-1, self.config.TIME_STEPS, self.config.n_outputs)
        outputs = self.X[:, self.config.TIME_STEPS -1, :]
        return outputs



