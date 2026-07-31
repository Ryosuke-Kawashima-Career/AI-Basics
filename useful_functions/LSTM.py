import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 1. Custom Embedding Layer
# ==========================================
class Embedding(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int):
        super().__init__()
        self.embedding_matrix = nn.Parameter(
            torch.randn((vocab_size, embedding_dim), dtype=torch.float)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.embedding(x, self.embedding_matrix)

# ==========================================
# 2. Custom Vanilla RNN Cell & Network
# ==========================================
class RNN(nn.Module):
    def __init__(self, batch_size: int, input_dim: int, output_dim: int, hidden_dim: int, vocab_size: int, embedding_length: int):
        super().__init__()
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.embedding_length = embedding_length
        
        # Xavier/Glorot uniform initialization
        limit = np.sqrt(6 / (self.hidden_dim + self.input_dim))
        self.W = nn.Parameter(torch.tensor(
            np.random.uniform(-limit, limit, size=(self.hidden_dim + self.input_dim, self.hidden_dim)).astype('float32')
        ))
        self.b = nn.Parameter(torch.zeros(self.hidden_dim, dtype=torch.float32))

    def activation(self, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Concatenates previous hidden state and input vector, applies linear transformation & tanh"""
        hx = torch.cat([h, x], dim=1)
        return torch.tanh(torch.matmul(hx, self.W) + self.b)

    def forward(self, x: torch.Tensor, max_len_seq: int = 0, init_state = None) -> torch.Tensor:
        # Transpose input: (batch, series, input_dim) -> (series, batch, input_dim)
        x = x.transpose(0, 1)
        batch_sz = x.size(1)
        
        state = init_state
        if state is None:
            state = torch.zeros((batch_sz, self.hidden_dim), device=x.device)
            
        if max_len_seq == 0:
            max_len_seq = x.size(0)  # Correct sequence length dimension (series)
            
        outputs = []
        for i in range(max_len_seq):
            state = self.activation(state, x[i])  # Step i along sequence: x[i] is (batch, input_dim)
            outputs.append(state.unsqueeze(0))
            
        return torch.cat(outputs, dim=0)  # Shape: (series, batch, hidden_dim)

# ==========================================
# 3. Custom LSTM Cell & Network
# ==========================================
class LSTM(nn.Module):
    def __init__(self, batch_size: int, input_dim: int, output_dim: int, hidden_dim: int):
        super().__init__()
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        
        limit = np.sqrt(6 / (self.hidden_dim * 2 + self.input_dim))
        
        # Gates initialization (Input, Forget, Output, Cell Candidate)
        self.W_in = nn.Parameter(torch.tensor(np.random.uniform(-limit, limit, (self.input_dim + self.hidden_dim, self.hidden_dim)).astype('float32')))
        self.b_in = nn.Parameter(torch.zeros(self.hidden_dim, dtype=torch.float32))
        
        self.W_forget = nn.Parameter(torch.tensor(np.random.uniform(-limit, limit, (self.input_dim + self.hidden_dim, self.hidden_dim)).astype('float32')))
        self.b_forget = nn.Parameter(torch.zeros(self.hidden_dim, dtype=torch.float32))
        
        self.W_out = nn.Parameter(torch.tensor(np.random.uniform(-limit, limit, (self.input_dim + self.hidden_dim, self.hidden_dim)).astype('float32')))
        self.b_out = nn.Parameter(torch.zeros(self.hidden_dim, dtype=torch.float32))
        
        self.W_cell = nn.Parameter(torch.tensor(np.random.uniform(-limit, limit, (self.input_dim + self.hidden_dim, self.hidden_dim)).astype('float32')))
        self.b_cell = nn.Parameter(torch.zeros(self.hidden_dim, dtype=torch.float32))

    def activation(self, c: torch.Tensor, h: torch.Tensor, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """LSTM Cell activation step returning (c_next, h_next)"""
        hx = torch.cat([h, x], dim=1)
        i_gate = torch.sigmoid(torch.matmul(hx, self.W_in) + self.b_in)
        f_gate = torch.sigmoid(torch.matmul(hx, self.W_forget) + self.b_forget)
        o_gate = torch.sigmoid(torch.matmul(hx, self.W_out) + self.b_out)
        g_cand = torch.tanh(torch.matmul(hx, self.W_cell) + self.b_cell)
        
        c_next = f_gate * c + i_gate * g_cand
        h_next = o_gate * torch.tanh(c_next)
        return c_next, h_next

    def forward(self, x: torch.Tensor, max_len_seq: int = 0, init_state = None) -> torch.Tensor:
        x = x.transpose(0, 1)  # (series, batch, features)
        batch_sz = x.size(1)
        
        if init_state is None:
            h = torch.zeros((batch_sz, self.hidden_dim), device=x.device)
            c = torch.zeros((batch_sz, self.hidden_dim), device=x.device)
        else:
            h, c = init_state

        if max_len_seq == 0:
            max_len_seq = x.size(0)

        outputs = []
        for i in range(max_len_seq):
            c, h = self.activation(c, h, x[i])
            outputs.append(h.unsqueeze(0))
            
        return torch.cat(outputs, dim=0)  # Shape: (series, batch, hidden_dim)

# ==========================================
# 4. Assembled Sequence Tagging / Classification Net
# ==========================================
class SequenceTaggingNet(nn.Module):
    def __init__(self, word_num: int, embed_dim: int, hidden_dim: int, use_lstm: bool = True):
        super(SequenceTaggingNet, self).__init__()
        self.emb = Embedding(word_num, embed_dim)
        self.use_lstm = use_lstm
        
        if use_lstm:
            self.rnn = nn.LSTM(embed_dim, hidden_dim, num_layers=1, batch_first=True)
        else:
            self.rnn = nn.RNN(embed_dim, hidden_dim, num_layers=1, batch_first=True)
            
        self.linear = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, seq_len_max: int = 0, seq_len: torch.Tensor = None, init_state = None) -> torch.Tensor:
        # 1. Look up embeddings
        h = self.emb(x)  # Shape: [Batch, Sequence, Embed_Dim]
        
        # 2. Slice sequence length if max length bound is set
        if seq_len_max > 0:
            h = h[:, 0:seq_len_max, :]
            
        # 3. Recurrent processing
        out, _ = self.rnn(h, init_state)  # Shape: [Batch, Sequence, Hidden_Dim]
        
        # Transpose to [Sequence, Batch, Hidden_Dim] for step selection
        out = out.transpose(0, 1)
        
        # 4. Extract last hidden state for static or dynamic sequence lengths
        if seq_len is not None:
            batch_size = out.size(1)
            # Pick hidden state at specific sequence length per batch element
            last_h = out[seq_len - 1, torch.arange(batch_size, device=x.device), :]
        else:
            # Take the final sequence step
            last_h = out[-1, :, :]
            
        # 5. Linear classification head
        y = self.linear(last_h)
        return y
