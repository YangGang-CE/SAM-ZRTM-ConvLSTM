import torch
import torch.nn as nn



class SelfAttentionMemoryModule(nn.Module):
    def __init__(self, input_dim, hidden_dim, device):
        super().__init__()
        
        # Attention layers for hidden state (h)
        self.query_layer = nn.Conv2d(input_dim, hidden_dim, 1)
        self.key_layer = nn.Conv2d(input_dim, hidden_dim, 1)
        self.value_layer = nn.Conv2d(input_dim, input_dim, 1)
        
        # Attention layers for memory state (m)
        self.key_layer_memory = nn.Conv2d(input_dim, hidden_dim, 1)
        self.value_layer_memory = nn.Conv2d(input_dim, input_dim, 1)
        
        # Layers for combining attention outputs
        self.combine_layer = nn.Conv2d(input_dim * 2, input_dim * 2, 1)
        self.memory_update_layer = nn.Conv2d(input_dim * 3, input_dim * 3, 1)
        
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim

    def forward(self, h, m):
        batch_size, channels, height, width = h.shape
        
        # Self-attention for hidden state
        K_h = self.key_layer(h)
        Q_h = self.query_layer(h)
        
        # Reshape for attention computation
        K_h = K_h.view(batch_size, self.hidden_dim, height * width)
        Q_h = Q_h.view(batch_size, self.hidden_dim, height * width)
        Q_h = Q_h.transpose(1, 2)  # [batch, H*W, hidden_dim]

        # Compute attention weights for hidden state
        A_h = torch.softmax(torch.bmm(Q_h, K_h), dim=-1)  # [batch, H*W, H*W]

        # Apply attention to hidden state values
        V_h = self.value_layer(h)
        V_h = V_h.view(batch_size, self.input_dim, height * width)
        Z_h = torch.matmul(A_h, V_h.permute(0, 2, 1))

        # Self-attention for memory state
        K_m = self.key_layer_memory(m)
        K_m = K_m.view(batch_size, self.hidden_dim, height * width)
        
        # Compute attention weights for memory (using hidden query)
        A_m = torch.softmax(torch.bmm(Q_h, K_m), dim=-1)
        
        # Apply attention to memory state values
        V_m = self.value_layer_memory(m)
        V_m = V_m.view(batch_size, self.input_dim, height * width)
        Z_m = torch.matmul(A_m, V_m.permute(0, 2, 1))
        
        # Reshape back to spatial dimensions
        Z_h = Z_h.transpose(1, 2).view(batch_size, self.input_dim, height, width)
        Z_m = Z_m.transpose(1, 2).view(batch_size, self.input_dim, height, width)

        # Combine attention outputs
        combined_attention = torch.cat([Z_h, Z_m], dim=1)
        Z = self.combine_layer(combined_attention)
        
        # Memory update mechanism
        memory_input = torch.cat([Z, h], dim=1)  # 3 * input_dim
        combined = self.memory_update_layer(memory_input)
        
        # Split into output, candidate, and input gates
        output_gate, candidate_gate, input_gate = torch.split(combined, self.input_dim, dim=1)
        
        # Apply gates for memory update
        input_gate = torch.sigmoid(input_gate)
        new_m = (1 - input_gate) * m + input_gate * torch.tanh(candidate_gate)
        new_h = torch.sigmoid(output_gate) * new_m

        return new_h, new_m



class SA_Convlstm_cell(nn.Module):
    def __init__(self, params):
        super().__init__()
        
        # Model hyperparameters
        self.input_channels = params['hidden_dim']  # Using hidden_dim due to encoder
        self.hidden_dim = params['hidden_dim']
        self.kernel_size = params['kernel_size']
        self.padding = params['padding']
        self.device = params['device']
        
        # Self-attention memory module
        self.attention_layer = SelfAttentionMemoryModule(
            params['hidden_dim'], 
            params['att_hidden_dim'], 
            self.device
        )
        
        # ConvLSTM convolution with group normalization
        self.conv2d = nn.Sequential(
            nn.Conv2d(in_channels=self.input_channels + self.hidden_dim, 
                     out_channels=4 * self.hidden_dim,
                     kernel_size=self.kernel_size, 
                     padding=self.padding),
            nn.GroupNorm(4 * self.hidden_dim, 4 * self.hidden_dim)
        )


    def forward(self, x, hidden):
        h, c, m = hidden
        
        # Concatenate input and hidden state
        combined = torch.cat([x, h], dim=1)
        
        # Apply ConvLSTM convolution
        combined_conv = self.conv2d(combined)
        
        # Split into gates
        input_gate, forget_gate, output_gate, candidate_gate = torch.split(
            combined_conv, self.hidden_dim, dim=1
        )
        
        # Apply activation functions
        input_gate = torch.sigmoid(input_gate)
        forget_gate = torch.sigmoid(forget_gate)
        output_gate = torch.sigmoid(output_gate)
        candidate_gate = torch.tanh(candidate_gate)
        
        # Update cell state
        c_next = forget_gate * c + input_gate * candidate_gate
        
        # Update hidden state
        h_next = output_gate * torch.tanh(c_next)
        
        # Apply self-attention mechanism
        h_next, m_next = self.attention_layer(h_next, m)

        return h_next, (h_next, c_next, m_next)

    def init_hidden(self, batch_size, img_size):
        height, width = img_size
        
        # Initialize all states with zeros
        h_0 = torch.zeros(batch_size, self.hidden_dim, height, width).to(self.device)
        c_0 = torch.zeros(batch_size, self.hidden_dim, height, width).to(self.device)
        m_0 = torch.zeros(batch_size, self.hidden_dim, height, width).to(self.device)
        
        return (h_0, c_0, m_0)