import torch
import torch.nn as nn
import random

from model.convLSTM_cell import ConvLSTMCell

class EncoderDecoderConvLSTM(nn.Module):
    def __init__(self, params):
        super(EncoderDecoderConvLSTM, self).__init__()

        # Model hyperparameters
        self.input_channels = params['hidden_dim']
        self.hidden_channels = params['hidden_dim']
        self.output_channels = params['hidden_dim']
        self.input_window_size = params['input_window_size']
        self.n_layers = params['n_layers']
        self.img_size = params['img_size']
        self.batch_size = params['batch_size']
        self.device = params['device']
        
        # Initialize ConvLSTM cells
        self.cells = []
        for i in range(self.n_layers):
            self.cells.append(ConvLSTMCell(in_channels=self.input_channels,
                                           h_channels=self.hidden_channels,
                                           kernel_size=(3, 3),
                                           bias=True))
        
        self.cells = nn.ModuleList(self.cells)
        
        # Image encoder: downsample input images (256x256 -> 16x16)
        self.img_encode = nn.Sequential(
            # 256x256 -> 256x256
            nn.Conv2d(in_channels=params['input_dim'], kernel_size=1, stride=1, padding=0,
                      out_channels=params['hidden_dim']),
            nn.LeakyReLU(0.1),
            # 256x256 -> 128x128
            nn.Conv2d(in_channels=params['hidden_dim'], kernel_size=3, stride=2, padding=1,
                      out_channels=params['hidden_dim']),
            nn.LeakyReLU(0.1),
            # 128x128 -> 64x64
            nn.Conv2d(in_channels=params['hidden_dim'], kernel_size=3, stride=2, padding=1,
                      out_channels=params['hidden_dim']),
            nn.LeakyReLU(0.1),
            # 64x64 -> 32x32
            nn.Conv2d(in_channels=params['hidden_dim'], kernel_size=3, stride=2, padding=1,
                      out_channels=params['hidden_dim']),
            nn.LeakyReLU(0.1),
            # 32x32 -> 16x16
            nn.Conv2d(in_channels=params['hidden_dim'], kernel_size=3, stride=2, padding=1,
                      out_channels=params['hidden_dim']),
            nn.LeakyReLU(0.1)
        )

        # Image decoder: upsample to original resolution (16x16 -> 256x256)
        self.img_decode = nn.Sequential(
            # 16x16 -> 32x32
            nn.ConvTranspose2d(in_channels=params['hidden_dim'], kernel_size=3, stride=2, 
                             padding=1, output_padding=1, out_channels=params['hidden_dim']),
            nn.LeakyReLU(0.1),
            # 32x32 -> 64x64
            nn.ConvTranspose2d(in_channels=params['hidden_dim'], kernel_size=3, stride=2, 
                             padding=1, output_padding=1, out_channels=params['hidden_dim']),
            nn.LeakyReLU(0.1),
            # 64x64 -> 128x128
            nn.ConvTranspose2d(in_channels=params['hidden_dim'], kernel_size=3, stride=2, 
                             padding=1, output_padding=1, out_channels=params['hidden_dim']),
            nn.LeakyReLU(0.1),
            # 128x128 -> 256x256
            nn.ConvTranspose2d(in_channels=params['hidden_dim'], kernel_size=3, stride=2, 
                             padding=1, output_padding=1, out_channels=params['hidden_dim']),
            nn.LeakyReLU(0.1),
            # Final output layer
            nn.Conv2d(in_channels=params['hidden_dim'], kernel_size=1, stride=1, padding=0,
                      out_channels=params['input_dim'])
        )

        # Decoder prediction layer
        self.decoder_predict = nn.Conv2d(in_channels=params['hidden_dim'],
                                         out_channels=params['hidden_dim'],
                                         kernel_size=(1, 1),
                                         padding=(0, 0))

    def forward(self, x, y, teacher_forcing_rate=0.5, hidden=None):
        batch_size, seq_len, x_channels, height, width = x.size()
        
        # Initialize hidden states if not provided
        if hidden is None:
            hidden = self.init_hidden(batch_size=batch_size, img_size=self.img_size)
        _, future_seq, y_channels, _, _ = y.size()

        # Concatenate input and target sequences
        frames = torch.cat([x, y], dim=1)
        predictions = []

        # Process each time step
        for t in range(19):  # Total sequence length (input + output)
            # Teacher forcing: use ground truth or previous prediction
            if t < self.input_window_size or random.random() < teacher_forcing_rate:
                current_frame = frames[:, t, :, :, :]
            else:
                current_frame = output
            
            # Encode current frame
            encoded_frame = self.img_encode(current_frame)

            # Pass through ConvLSTM layers
            for layer_idx, cell in enumerate(self.cells):
                hidden[layer_idx] = cell(input_tensor=encoded_frame, cur_state=hidden[layer_idx])
                
                if layer_idx == 0:
                    output = self.decoder_predict(hidden[layer_idx][0])
                else:
                    output = self.decoder_predict(hidden[layer_idx][0])

            # Decode output frame
            output = self.img_decode(output)
            predictions.append(output)

        # Stack predictions and return only the prediction part
        predictions = torch.stack(predictions, dim=1)
        final_predictions = predictions[:, 9:, :, :, :]  # Return only predicted frames

        return final_predictions


    def init_hidden(self, batch_size, img_size):
        states = []
        for i in range(self.n_layers):
            states.append(self.cells[i].init_hidden(batch_size, img_size))
        return states

