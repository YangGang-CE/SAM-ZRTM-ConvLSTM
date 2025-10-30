import torch
import torch.nn as nn
import random

from model.sa_convLSTM_cell import SA_Convlstm_cell

class Encode2Decode(nn.Module):

    def __init__(self, params):

        super(Encode2Decode, self).__init__()
        
        # Model hyperparameters
        self.batch_size = params['batch_size']
        self.img_size = params['img_size']
        self.n_layers = params['n_layers']
        self.input_window_size = params['input_window_size']
        self.output_window_size = params['output_dim']
        
        # Initialize cell and normalization lists
        self.cells = []
        self.batch_norms = []

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

        # Calculate the feature map size after encoding
        # Four stride=2 convolutions: input_size -> input_size/16
        encoded_height = params['img_size'][0] // 16
        encoded_width = params['img_size'][1] // 16
        
        print(f"Model initialized with:")
        print(f"  Input image size: {params['img_size']}")
        print(f"  Encoded feature size: ({encoded_height}, {encoded_width})")
        
        # Create SA-ConvLSTM cells and layer normalization
        for i in range(params['n_layers']):
            self.cells.append(SA_Convlstm_cell(params))
            self.batch_norms.append(nn.LayerNorm((params['hidden_dim'], encoded_height, encoded_width)))

        # Convert to ModuleList for proper parameter registration
        self.cells = nn.ModuleList(self.cells)
        self.batch_norms = nn.ModuleList(self.batch_norms)

        # Prediction layer (currently unused)
        self.decoder_predict = nn.Conv2d(in_channels=params['hidden_dim'],
                                         out_channels=1,
                                         kernel_size=(1, 1),
                                         padding=(0, 0))

    def forward(self, x, y, teacher_forcing_rate=0.5, hidden=None):

        batch_size, seq_len, x_channels, height, width = x.size()
        
        if hidden is None:
            hidden = self.init_hidden(batch_size=batch_size, img_size=self.img_size)
        _, horizon, y_channels, _, _ = y.size()

        predictions = []
        
        # Concatenate input and target sequences
        frames = torch.cat([x, y], dim=1)
        
        # Memory state for zigzag mechanism
        memory_state = None

        # Process each time step
        for t in range(19):  # Total sequence length (input + output)
            # Teacher forcing: use ground truth or previous prediction
            if t < self.input_window_size or random.random() < teacher_forcing_rate:
                current_frame = frames[:, t, :, :, :]
            else:
                current_frame = output

            # Encode current frame
            encoded_frame = self.img_encode(current_frame)

            # Pass through SA-ConvLSTM layers with zigzag mechanism
            for layer_idx, cell in enumerate(self.cells):
                if layer_idx == 0 and t == 0:
                    # First layer, first time step
                    output, hidden[layer_idx] = cell(encoded_frame, 
                                                   (hidden[layer_idx][0], 
                                                    hidden[layer_idx][1], 
                                                    hidden[layer_idx][2]))
                    output = self.batch_norms[layer_idx](output)
                    
                elif layer_idx == 0 and t != 0:
                    # First layer, subsequent time steps
                    output, hidden[layer_idx] = cell(encoded_frame,
                                                   (hidden[layer_idx][0], 
                                                    hidden[layer_idx][1], 
                                                    memory_state))
                    output = self.batch_norms[layer_idx](output)
                else:
                    # Higher layers: use memory from previous layer
                    output, hidden[layer_idx] = cell(encoded_frame, 
                                                   (hidden[layer_idx][0], 
                                                    hidden[layer_idx][1], 
                                                    hidden[layer_idx-1][2]))
                    output = self.batch_norms[layer_idx](output)
                    memory_state = hidden[layer_idx][2]

            # Decode output frame
            output = self.img_decode(output)
            predictions.append(output)

        # Stack predictions and return only the prediction part
        predictions = torch.stack(predictions, dim=1)
        predictions = predictions[:, 9:, :, :, :]  # Return only predicted frames

        return predictions


    def init_hidden(self, batch_size, img_size):
W
        # Calculate the encoded feature map size
        # Four stride=2 convolutions: input_size -> input_size/16
        encoded_height = img_size[0] // 16
        encoded_width = img_size[1] // 16
        encoded_size = (encoded_height, encoded_width)
        
        states = []
        for i in range(self.n_layers):
            states.append(self.cells[i].init_hidden(batch_size, encoded_size))
        return states