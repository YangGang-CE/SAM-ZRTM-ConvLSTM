import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from model.Encode2Decode import Encode2Decode
from model.seq2seq import EncoderDecoderConvLSTM


class Predictor:
    def __init__(self, model_path: str, model_type: str = 'sa_convlstm', device: str = 'auto', 
                 config_path: str = None):
        """
        Initialize the predictor.
        
        Args:
            model_path (str): Path to the trained model weights
            model_type (str): Type of model ('sa_convlstm' or 'convlstm')
            device (str): Device to use ('auto', 'cpu', 'cuda')
            config_path (str, optional): Path to training config JSON file for automatic parameter loading
        """
        self.model_path = model_path
        self.model_type = model_type
        self.config_path = config_path
        
        # Set device
        if device == 'auto':
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Load parameters from config file if provided, otherwise use defaults
        self.params = self._load_model_params()
        
        # Ensure dropout is disabled for inference
        self.params['dropout_rate'] = 0.0
        print(f"Dropout disabled for inference (dropout_rate=0.0)")
        
        # Load model
        self.model = self._load_model()
        
        # Initialize Monte Carlo Dropout predictor if available
        self.monte_carlo_predictor = None
        self._initialize_monte_carlo_predictor()
    
    def _load_model_params(self):
        """Load model parameters from config file or use defaults."""
        if self.config_path and os.path.exists(self.config_path):
            print(f"Loading parameters from config: {self.config_path}")
            import json
            try:
                with open(self.config_path, 'r') as f:
                    config_data = json.load(f)
                
                # Extract model parameters from config
                model_params = config_data.get('model_params', {})
                optimization_params = config_data.get('optimization_params', {})
                
                params = {
                    'input_dim': 1,
                    'batch_size': 1,  # Single prediction
                    'padding': 1,
                    'lr': 1e-3,
                    'device': self.device,
                    'att_hidden_dim': model_params.get('attention_hidden_dim', 64),
                    'kernel_size': 3,
                    'img_size': (256, 256),  # Default, can be overridden
                    'hidden_dim': model_params.get('hidden_dim', 64),
                    'n_layers': model_params.get('num_layers', 4),
                    'output_dim': 10,
                    'input_window_size': 10,
                    'loss': 'L2',
                    'model_cell': self.model_type,
                    'bias': model_params.get('bias', True),
                    'dropout_rate': optimization_params.get('dropout_rate', 0.0)
                }
                
                print(f"Loaded config: hidden_dim={params['hidden_dim']}, "
                      f"n_layers={params['n_layers']}, "
                      f"att_hidden_dim={params['att_hidden_dim']}")
                
                return params
                
            except Exception as e:
                print(f"Failed to load config file: {e}")
                print("Using default parameters...")
        
        # Auto-detect parameters from checkpoint
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)
            hidden_dim = self._detect_hidden_dim_from_checkpoint(checkpoint)
            n_layers = self._detect_n_layers_from_checkpoint(checkpoint)
            print(f"Auto-detected from checkpoint: hidden_dim={hidden_dim}, n_layers={n_layers}")
        except Exception as e:
            print(f"Could not auto-detect parameters: {e}")
            hidden_dim = 64  # Safe default
            n_layers = 4   # Safe default
        
        # Default parameters (auto-detected from checkpoint)
        return {
            'input_dim': 1,
            'batch_size': 1,  # Single prediction
            'padding': 1,
            'lr': 1e-3,
            'device': self.device,
            'att_hidden_dim': hidden_dim,  # Auto-detected
            'kernel_size': 3,
            'img_size': (256, 256),
            'hidden_dim': hidden_dim,  # Auto-detected
            'n_layers': n_layers,  # Auto-detected
            'output_dim': 10,
            'input_window_size': 10,
            'loss': 'L2',
            'model_cell': self.model_type,
            'bias': True,
            'dropout_rate': 0.0
        }
        
    def _load_model(self):
        """Load the trained model with dropout compatibility."""
        print(f"Loading model from: {self.model_path}")
        
        # Create model
        if self.model_type == 'sa_convlstm':
            model = Encode2Decode(self.params).to(self.device)
        else:
            model = EncoderDecoderConvLSTM(self.params).to(self.device)
        
        # Load weights
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        try:
            state_dict = torch.load(self.model_path, map_location=self.device)
            
            # Handle potential compatibility issues with dropout layers
            model_state_dict = model.state_dict()
            
            # Check for missing keys (e.g., dropout layers in new model but not in saved model)
            missing_keys = set(model_state_dict.keys()) - set(state_dict.keys())
            unexpected_keys = set(state_dict.keys()) - set(model_state_dict.keys())
            
            if missing_keys:
                print(f"Missing keys in saved model (will use default initialization): {missing_keys}")
            
            if unexpected_keys:
                print(f"Unexpected keys in saved model (will be ignored): {unexpected_keys}")
            
            # Load state dict with strict=False to handle dropout layer compatibility
            model.load_state_dict(state_dict, strict=False)
            model.eval()
            
            # Verify dropout layers are in eval mode
            dropout_layers = []
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Dropout2d):
                    dropout_layers.append(name)
            
            if dropout_layers:
                print(f"Found dropout layers (disabled in eval mode): {dropout_layers}")
            else:
                print("No dropout layers found in model")
            
            print("Model loaded successfully!")
            return model
            
        except Exception as e:
            print(f"Error loading model: {e}")
            print("This might be due to model architecture mismatch.")
            print("Please check that the model parameters match the training configuration.")
            raise
    
    def load_image_sequence(self, image_paths: list) -> torch.Tensor:
        """
        Load a sequence of images from file paths.
        
        Args:
            image_paths (list): List of paths to image files
        
        Returns:
            torch.Tensor: Tensor with shape [seq_len, channels, height, width]
        """
        if len(image_paths) != 10:
            raise ValueError(f"Expected 10 images, got {len(image_paths)}")
        
        frames = []
        target_size = self.params['img_size']
        
        for img_path in image_paths:
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Image not found: {img_path}")
            
            # Load and preprocess image
            img = Image.open(img_path).convert('L')  # Convert to grayscale
            
            # Resize if necessary
            if img.size != target_size:
                img = img.resize(target_size, Image.LANCZOS)
            
            # Convert to numpy array and normalize
            frame_array = np.array(img, dtype=np.float32) / 255.0
            frames.append(frame_array)
        
        # Stack frames and add channel dimension
        sequence_array = np.stack(frames, axis=0)  # [seq_len, height, width]
        sequence_array = np.expand_dims(sequence_array, axis=1)  # [seq_len, channels, height, width]
        
        # Convert to tensor
        sequence_tensor = torch.from_numpy(sequence_array).to(self.device)
        
        return sequence_tensor
    
    def load_sequence_from_directory(self, sequence_dir: str) -> torch.Tensor:
        """
        Load a sequence from a directory containing numbered images.
        
        Args:
            sequence_dir (str): Directory containing images named 00.png, 01.png, ..., 09.png
        
        Returns:
            torch.Tensor: Input sequence tensor
        """
        image_paths = []
        for i in range(10):
            img_path = os.path.join(sequence_dir, f"{i:02d}.png")
            image_paths.append(img_path)
        
        return self.load_image_sequence(image_paths)
    
    def predict(self, input_sequence: torch.Tensor) -> torch.Tensor:
        """
        Make prediction on input sequence.
        
        Args:
            input_sequence (torch.Tensor): Input sequence [seq_len, channels, height, width]
        
        Returns:
            torch.Tensor: Predicted sequence [horizon, channels, height, width]
        """
        # Add batch dimension
        input_batch = input_sequence.unsqueeze(0)  # [1, seq_len, channels, height, width]
        
        # Create dummy target (not used in prediction mode)
        dummy_target = torch.zeros_like(input_batch)  # Same shape as input
        
        with torch.no_grad():
            # Make prediction (teacher_forcing_rate=0 means no teacher forcing)
            prediction = self.model(input_batch, dummy_target, teacher_forcing_rate=0.0)
        
        # Remove batch dimension
        prediction = prediction.squeeze(0)  # [horizon, channels, height, width]
        
        return prediction
    
    def _detect_hidden_dim_from_checkpoint(self, checkpoint: dict) -> int:
        """
        Auto-detect hidden_dim from checkpoint by examining layer shapes.
        
        Args:
            checkpoint: Model state dict
            
        Returns:
            Detected hidden dimension
        """
        # Check img_encode.0.weight shape to determine hidden_dim
        if 'img_encode.0.weight' in checkpoint:
            # Shape should be [hidden_dim, input_dim, 1, 1]
            hidden_dim = checkpoint['img_encode.0.weight'].shape[0]
            return hidden_dim
        
        # Fallback: check cells.0 attention layer
        if 'cells.0.attention_layer.query_layer.weight' in checkpoint:
            # Shape should be [hidden_dim, hidden_dim, 1, 1]
            hidden_dim = checkpoint['cells.0.attention_layer.query_layer.weight'].shape[0]
            return hidden_dim
        
        # Default fallback
        return 64
    
    def _detect_n_layers_from_checkpoint(self, checkpoint: dict) -> int:
        """
        Auto-detect number of layers from checkpoint.
        
        Args:
            checkpoint: Model state dict
            
        Returns:
            Detected number of layers
        """
        # Count cells.X.* keys to determine number of layers
        layer_indices = set()
        for key in checkpoint.keys():
            if key.startswith('cells.'):
                # Extract layer index from keys like 'cells.0.attention_layer.query_layer.weight'
                parts = key.split('.')
                if len(parts) >= 2 and parts[1].isdigit():
                    layer_indices.add(int(parts[1]))
        
        if layer_indices:
            n_layers = max(layer_indices) + 1  # +1 because indices start from 0
            return n_layers
        
        # Fallback: check batch_norms
        batch_norm_indices = set()
        for key in checkpoint.keys():
            if key.startswith('batch_norms.'):
                parts = key.split('.')
                if len(parts) >= 2 and parts[1].isdigit():
                    batch_norm_indices.add(int(parts[1]))
        
        if batch_norm_indices:
            n_layers = max(batch_norm_indices) + 1
            return n_layers
        
        # Default fallback
        return 4
    
    def _initialize_monte_carlo_predictor(self):
        """Initialize Monte Carlo Dropout predictor if available."""
        if not MONTE_CARLO_AVAILABLE:
            print("Monte Carlo Dropout not available - ensemble generation will use fallback method")
            return
        
        try:
            # Create Monte Carlo configuration
            mc_config = {
                'model_type': self.model_type,
                'ensemble_size': 50,  # Default ensemble size
                'dropout_rate': 0.1,  # Default dropout rate for Monte Carlo
                'seed': None,
                'temperature': 1.0,
                'batch_processing': True
            }
            
            # Initialize Monte Carlo predictor
            self.monte_carlo_predictor = MonteCarloPredictor(
                model_path=self.model_path,
                config=mc_config
            )
            
            print("✅ Monte Carlo Dropout predictor initialized successfully")
            
        except Exception as e:
            print(f"Failed to initialize Monte Carlo Dropout predictor: {str(e)}")
            print("Ensemble generation will use fallback method")
            self.monte_carlo_predictor = None
    
    def save_predictions(self, predictions: torch.Tensor, output_dir: str, prefix: str = "pred"):
        """
        Save predicted images to directory.
        
        Args:
            predictions (torch.Tensor): Predicted sequence [horizon, channels, height, width]
            output_dir (str): Output directory
            prefix (str): Prefix for output files
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert to numpy and denormalize
        pred_numpy = predictions.cpu().numpy()
        pred_numpy = np.clip(pred_numpy * 255.0, 0, 255).astype(np.uint8)
        
        saved_files = []
        for i in range(pred_numpy.shape[0]):
            frame = pred_numpy[i, 0]  # Remove channel dimension
            
            # Save as PNG
            img = Image.fromarray(frame, mode='L')
            filename = f"{prefix}_{i:02d}.png"
            filepath = os.path.join(output_dir, filename)
            img.save(filepath)
            saved_files.append(filepath)
        
        print(f"Saved {len(saved_files)} predicted images to: {output_dir}")
        return saved_files
    
    def visualize_prediction(self, input_sequence: torch.Tensor, predictions: torch.Tensor, 
                           save_path: str = None, show: bool = True):
        """
        Visualize input sequence and predictions.
        
        Args:
            input_sequence (torch.Tensor): Input sequence
            predictions (torch.Tensor): Predicted sequence
            save_path (str, optional): Path to save visualization
            show (bool): Whether to display the plot
        """
        # Convert to numpy
        input_numpy = input_sequence.cpu().numpy()
        pred_numpy = predictions.cpu().numpy()
        
        # Create visualization
        fig, axes = plt.subplots(2, 10, figsize=(20, 6))
        fig.suptitle('MovingMNIST Sequence Prediction', fontsize=16)
        
        # Plot input sequence
        for i in range(10):
            frame = input_numpy[i, 0]  # Remove channel dimension
            axes[0, i].imshow(frame, cmap='gray', vmin=0, vmax=1)
            axes[0, i].set_title(f'Input {i+1}')
            axes[0, i].axis('off')
        
        # Plot predicted sequence
        for i in range(10):
            frame = pred_numpy[i, 0]  # Remove channel dimension
            axes[1, i].imshow(frame, cmap='gray', vmin=0, vmax=1)
            axes[1, i].set_title(f'Pred {i+1}')
            axes[1, i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()

