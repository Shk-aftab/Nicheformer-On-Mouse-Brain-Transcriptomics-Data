"""
Nicheformer Model Wrapper - Milestone 3

Wraps the Nicheformer model to provide a clean, standardized interface
for centralized and federated training.

Contract Reference: docs/milestone3/contracts/model_contract.md
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, List
import os


class NicheformerWrapper(nn.Module):
    """
    Wrapper for Nicheformer model that implements the Model Contract interface.
    
    This wrapper provides:
    - forward(batch) -> logits
    - compute_loss(logits, labels) -> scalar
    - get_weights() -> state_dict
    - set_weights(state_dict)
    - Configurable fine-tuning modes (head-only, partial, full)
    """
    
    def __init__(
        self,
        num_genes: int,
        num_labels: int,
        pretrained_path: Optional[str] = None,
        fine_tune_mode: str = "head_only",
        include_spatial: bool = True,
        hidden_dim: int = 256,
        num_layers: int = 4,
        dropout: float = 0.1
    ):
        """
        Initialize Nicheformer wrapper.
        
        Args:
            num_genes: Number of input genes (248 for Xenium)
            num_labels: Number of output classes (22 for Leiden clusters)
            pretrained_path: Path to pretrained Nicheformer weights (optional)
            fine_tune_mode: One of "head_only", "partial", "full"
            include_spatial: Whether to include spatial coordinates (x, y) in input
            hidden_dim: Hidden dimension for model (if not using pretrained)
            num_layers: Number of transformer layers (if not using pretrained)
            dropout: Dropout rate
        """
        super().__init__()
        
        self.num_genes = num_genes
        self.num_labels = num_labels
        self.fine_tune_mode = fine_tune_mode
        self.include_spatial = include_spatial
        
        # Input dimension: genes + optionally spatial coords (x, y)
        input_dim = num_genes + (2 if include_spatial else 0)
        
        # Try to load pretrained Nicheformer if available
        self.backbone = self._load_backbone(pretrained_path, input_dim, hidden_dim, num_layers)
        
        # Track if we're using actual Nicheformer
        self._using_nicheformer = getattr(self, '_using_nicheformer', False)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_labels)
        )
        
        # Configure fine-tuning mode
        self._configure_fine_tuning()
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
    
    def _load_backbone(self, pretrained_path: Optional[str], input_dim: int, hidden_dim: int, num_layers: int) -> nn.Module:
        """
        Load Nicheformer backbone or create a simple transformer-based backbone.
        
        If pretrained Nicheformer is available, use it. Otherwise, create a simple
        transformer encoder as a placeholder.
        
        Args:
            pretrained_path: Path to pretrained Nicheformer checkpoint (.ckpt file)
            input_dim: Input dimension (genes + spatial coords)
            hidden_dim: Hidden dimension for model
            num_layers: Number of transformer layers
            
        Returns:
            Backbone model (Nicheformer or placeholder)
        """
        # Try to load actual Nicheformer if available
        try:
            import nicheformer
            from nicheformer.models import Nicheformer
            import pytorch_lightning as pl
            
            if pretrained_path and os.path.exists(pretrained_path):
                print(f"Loading pretrained Nicheformer from {pretrained_path}")
                # Load pretrained Nicheformer model
                nicheformer_model = Nicheformer.load_from_checkpoint(pretrained_path)
                # Extract the transformer encoder backbone
                backbone = nicheformer_model.encoder
                # Store embeddings for later use if needed
                self._nicheformer_embeddings = nicheformer_model.embeddings
                self._nicheformer_pos_embedding = nicheformer_model.positional_embedding
                self._nicheformer_dropout = getattr(nicheformer_model, 'dropout', None)
                self._nicheformer_pos = getattr(nicheformer_model, 'pos', None)
                self._using_nicheformer = True
                print("Successfully loaded Nicheformer backbone")
                return backbone
            else:
                # Create new Nicheformer model (without pretrained weights)
                print("Creating new Nicheformer model (no pretrained weights)")
                # Use default Nicheformer hyperparameters
                nicheformer_model = Nicheformer(
                    dim_model=hidden_dim,
                    nheads=8,
                    dim_feedforward=hidden_dim * 4,
                    nlayers=num_layers,
                    dropout=0.1,
                    batch_first=True,
                    masking_p=0.0,  # No masking for fine-tuning
                    n_tokens=1000,  # Will be adjusted based on actual vocabulary
                    context_length=512,  # Adjust based on your data
                    lr=1e-4,
                    warmup=1000,
                    batch_size=32,
                    max_epochs=10
                )
                backbone = nicheformer_model.encoder
                self._nicheformer_embeddings = nicheformer_model.embeddings
                self._nicheformer_pos_embedding = nicheformer_model.positional_embedding
                self._nicheformer_dropout = getattr(nicheformer_model, 'dropout', None)
                self._nicheformer_pos = getattr(nicheformer_model, 'pos', None)
                self._using_nicheformer = True
                print("Created new Nicheformer model")
                return backbone
                
        except ImportError:
            # Nicheformer not available, use placeholder
            print("Nicheformer package not available. Using placeholder backbone.")
            print("To use actual Nicheformer:")
            print("  1. Install from: https://github.com/theislab/nicheformer/")
            print("  2. Download pretrained weights from Mendeley Data")
            print("  3. Provide pretrained_path when creating model")
            self._using_nicheformer = False
            return self._create_placeholder_backbone(input_dim, hidden_dim, num_layers)
        except Exception as e:
            # Error loading Nicheformer, fall back to placeholder
            print(f"Error loading Nicheformer: {e}")
            print("Falling back to placeholder backbone.")
            self._using_nicheformer = False
            return self._create_placeholder_backbone(input_dim, hidden_dim, num_layers)
    
    def _create_placeholder_backbone(self, input_dim: int, hidden_dim: int, num_layers: int) -> nn.Module:
        """
        Create a simple transformer encoder as placeholder for Nicheformer.
        
        This is a minimal implementation for testing. In production, this should
        be replaced with the actual Nicheformer model from:
        https://github.com/theislab/nicheformer/
        """
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        
        # Project input to hidden dimension
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        return nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    
    def _configure_fine_tuning(self):
        """
        Configure which parameters are trainable based on fine_tune_mode.
        
        - head_only: Only classifier head is trainable
        - partial: Last N layers of backbone + head are trainable
        - full: All parameters are trainable
        """
        if self.fine_tune_mode == "head_only":
            # Freeze backbone
            for param in self.backbone.parameters():
                param.requires_grad = False
            # Unfreeze classifier
            for param in self.classifier.parameters():
                param.requires_grad = True
                
        elif self.fine_tune_mode == "partial":
            # Freeze early layers, unfreeze later layers
            if hasattr(self.backbone, 'layers'):
                # If backbone has layers attribute
                num_layers = len(self.backbone.layers)
                for i, layer in enumerate(self.backbone.layers):
                    if i < num_layers // 2:
                        # Freeze first half
                        for param in layer.parameters():
                            param.requires_grad = False
                    else:
                        # Unfreeze second half
                        for param in layer.parameters():
                            param.requires_grad = True
            # Always unfreeze classifier
            for param in self.classifier.parameters():
                param.requires_grad = True
                
        elif self.fine_tune_mode == "full":
            # Unfreeze everything
            for param in self.parameters():
                param.requires_grad = True
        else:
            raise ValueError(f"Unknown fine_tune_mode: {self.fine_tune_mode}. Must be 'head_only', 'partial', or 'full'")
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            batch: Dictionary containing:
                - 'features': Tensor of shape (batch_size, num_genes + 2) if include_spatial
                  or (batch_size, num_genes) otherwise
                - 'label': Tensor of shape (batch_size,) - not used in forward, only for loss
        
        Returns:
            logits: Tensor of shape (batch_size, num_labels)
        """
        features = batch['features']  # (batch_size, input_dim)
        
        if self._using_nicheformer:
            # Using actual Nicheformer backbone
            # Nicheformer expects tokenized input, but we have continuous features
            # Project to hidden dimension and treat as sequence
            if hasattr(self, 'input_projection'):
                x = self.input_projection(features)  # (batch_size, hidden_dim)
            else:
                # Create projection if not exists
                if not hasattr(self, '_temp_projection'):
                    self._temp_projection = nn.Linear(features.shape[1], self.backbone.layers[0].self_attn.embed_dim).to(features.device)
                x = self._temp_projection(features)
            
            # Add sequence dimension for transformer (treat each cell as a sequence of length 1)
            x = x.unsqueeze(1)  # (batch_size, 1, hidden_dim)
            
            # Pass through Nicheformer encoder
            # Nicheformer encoder expects (batch, seq_len, hidden_dim) and optional mask
            x = self.backbone(x, is_causal=False)  # (batch_size, 1, hidden_dim)
            
            # Remove sequence dimension
            x = x.squeeze(1)  # (batch_size, hidden_dim)
        else:
            # Using placeholder backbone
            # Project input to hidden dimension
            if hasattr(self, 'input_projection'):
                x = self.input_projection(features)  # (batch_size, hidden_dim)
            else:
                x = features
            
            # Add sequence dimension for transformer (treat each cell as a sequence of length 1)
            x = x.unsqueeze(1)  # (batch_size, 1, hidden_dim)
            
            # Pass through backbone
            x = self.backbone(x)  # (batch_size, 1, hidden_dim)
            
            # Remove sequence dimension
            x = x.squeeze(1)  # (batch_size, hidden_dim)
        
        # Classification head
        logits = self.classifier(x)  # (batch_size, num_labels)
        
        return logits
    
    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute classification loss.
        
        Args:
            logits: Model output logits, shape (batch_size, num_labels)
            labels: Ground truth labels, shape (batch_size,)
            
        Returns:
            loss: Scalar loss value
        """
        return self.criterion(logits, labels)
    
    def get_weights(self) -> Dict[str, torch.Tensor]:
        """
        Get model state dictionary for federated aggregation.
        
        Returns:
            state_dict: Dictionary mapping parameter names to tensors
        """
        return self.state_dict()
    
    def set_weights(self, state_dict: Dict[str, torch.Tensor]):
        """
        Set model weights from state dictionary (used in federated learning).
        
        Args:
            state_dict: Dictionary mapping parameter names to tensors
        """
        self.load_state_dict(state_dict, strict=False)
    
    def get_trainable_parameters(self) -> List[torch.nn.Parameter]:
        """
        Get list of trainable parameters (useful for optimizer setup).
        
        Returns:
            List of parameters that require gradients
        """
        return [p for p in self.parameters() if p.requires_grad]
    
    def count_parameters(self) -> Tuple[int, int]:
        """
        Count total and trainable parameters.
        
        Returns:
            (total_params, trainable_params)
        """
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


def create_model(
    num_genes: int,
    num_labels: int,
    pretrained_path: Optional[str] = None,
    fine_tune_mode: str = "head_only",
    include_spatial: bool = True,
    **kwargs
) -> NicheformerWrapper:
    """
    Factory function to create a NicheformerWrapper model.
    
    Args:
        num_genes: Number of input genes
        num_labels: Number of output classes
        pretrained_path: Path to pretrained weights
        fine_tune_mode: Fine-tuning mode ("head_only", "partial", "full")
        include_spatial: Whether to include spatial coordinates
        **kwargs: Additional arguments passed to NicheformerWrapper
        
    Returns:
        Initialized NicheformerWrapper model
    """
    return NicheformerWrapper(
        num_genes=num_genes,
        num_labels=num_labels,
        pretrained_path=pretrained_path,
        fine_tune_mode=fine_tune_mode,
        include_spatial=include_spatial,
        **kwargs
    )
