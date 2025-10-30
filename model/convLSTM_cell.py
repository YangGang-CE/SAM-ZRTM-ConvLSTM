import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):

    def __init__(self, in_channels, h_channels, kernel_size, bias=True):

        super(ConvLSTMCell, self).__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.h_channels = h_channels
        
        # Calculate padding to maintain spatial dimensions
        padding = kernel_size[0] // 2, kernel_size[1] // 2

        # Convolution for input-to-hidden and hidden-to-hidden transformations
        self.conv = nn.Conv2d(in_channels=in_channels + h_channels,
                              out_channels=4 * h_channels,
                              kernel_size=kernel_size,
                              padding=padding,
                              bias=bias)
        
    def forward(self, input_tensor, cur_state):

        h_cur, c_cur = cur_state
        
        # Concatenate input and hidden state along channel dimension
        combined = torch.cat([input_tensor, h_cur], dim=1)

        # Apply convolution to get all gate values
        combined_conv = self.conv(combined)
        
        # Split into input, forget, output, and candidate gates
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.h_channels, dim=1)
        
        # Apply activation functions
        input_gate = torch.sigmoid(cc_i)
        forget_gate = torch.sigmoid(cc_f)
        output_gate = torch.sigmoid(cc_o)
        candidate_gate = torch.tanh(cc_g)

        # Update cell state
        c_next = forget_gate * c_cur + input_gate * candidate_gate
        
        # Update hidden state
        h_next = output_gate * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):

        height, width = image_size
        
        # Initialize hidden and cell states with zeros
        h_0 = torch.zeros(batch_size, self.h_channels, height, width).to(self.device)
        c_0 = torch.zeros(batch_size, self.h_channels, height, width).to(self.device)
        
        return (h_0, c_0)

