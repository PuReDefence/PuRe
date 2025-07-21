import math
from typing import Union, Tuple, Optional

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    BertModel,
    BertPreTrainedModel,
)
from transformers.modeling_outputs import (
    SequenceClassifierOutput, 
    MultipleChoiceModelOutput, 
    QuestionAnsweringModelOutput
)
from transformers.utils import ModelOutput


class PFSA(torch.nn.Module):
    """
    Parameter-free Self-attention (PFSA) Module
    
    Implements the attention mechanism described in:
    https://openreview.net/pdf?id=isodM5jTA7h
    """
    
    def __init__(self, input_dim: int, alpha: float = 1.0):
        """
        Initialize PFSA module.
        
        Args:
            input_dim (int): Dimension of input features
            alpha (float): Scaling factor for attention weights. Default: 1.0
        """
        super(PFSA, self).__init__()
        self.input_dim = input_dim
        self.alpha = alpha

    def forward_one_sample(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for a single sample.
        
        Args:
            x (torch.Tensor): Input tensor of shape [1, T, F]
            
        Returns:
            torch.Tensor: Output tensor with applied attention
        """
        # Transpose and add extra dimension: [1, F, T, 1]
        x = x.transpose(1, 2)[..., None]
        
        # Compute mean key representation: [1, F, 1, 1]
        k = torch.mean(x, dim=[-1, -2], keepdim=True)
        
        # Compute standard deviation of key: [1, 1, 1, 1]
        k_centered = k - k.mean(dim=1, keepdim=True)
        kd = torch.sqrt(k_centered.pow(2).sum(dim=1, keepdim=True))
        
        # Compute standard deviation of query: [1, 1, T, 1]
        x_centered = x - x.mean(dim=1, keepdim=True)
        qd = torch.sqrt(x_centered.pow(2).sum(dim=1, keepdim=True))
        
        # Compute correlation coefficient between query and key
        correlation_numerator = (x_centered * k_centered).sum(dim=1, keepdim=True)
        C_qk = correlation_numerator / (qd * kd)
        
        # Compute attention weights using inverse sigmoid
        A = (1 - torch.sigmoid(C_qk)) ** self.alpha
        
        # Apply attention and reshape output
        out = x * A
        out = out.squeeze(dim=-1).transpose(1, 2)
        
        return out

    def forward(
        self, 
        input_values: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for batch of samples.
        
        Args:
            input_values (torch.Tensor): Input tensor of shape [B, T, F]
            attention_mask (torch.Tensor, optional): Attention mask of shape [B, T]
            
        Returns:
            torch.Tensor: Output tensor of shape [B, T, F]
        """
        batch_size, seq_len, feature_dim = input_values.shape
        output_list = []
        
        # Process each sample in the batch
        for sample, mask in zip(input_values, attention_mask):
            sample = sample.view(1, seq_len, feature_dim)
            
            # Apply mask to get valid sequence length
            valid_length = int(mask.sum().item())
            masked_sample = sample[:, :valid_length, :]
            
            # Apply attention to the masked sample
            attended_sample = self.forward_one_sample(masked_sample)
            
            # Pad back to original sequence length
            padded_sample = torch.zeros_like(sample, device=sample.device)
            padded_sample[:, :attended_sample.shape[1], :] = attended_sample
            
            output_list.append(padded_sample)
        
        # Stack all processed samples
        output = torch.vstack(output_list)
        output = output.view(batch_size, seq_len, feature_dim)
        
        return output


class PURE(torch.nn.Module):
    """
    Purified Representation (PURE) Module
    
    Combines Principal Component Removal (PCR) with Parameter-free 
    Self-attention (PFSA) for enhanced feature representation.
    """
    
    def __init__(
        self, 
        in_dim: int, 
        target_rank: int = 5, 
        npc: int = 1, 
        center: bool = False, 
        num_iters: int = 1, 
        alpha: float = 1.0, 
        do_pcr: bool = True, 
        do_pfsa: bool = True, 
        *args, 
        **kwargs
    ):
        """
        Initialize PURE module.
        
        Args:
            in_dim (int): Input dimension
            target_rank (int): Target rank for PCA. Default: 5
            npc (int): Number of principal components to remove. Default: 1
            center (bool): Whether to center data in PCA. Default: False
            num_iters (int): Number of iterations for PCA. Default: 1
            alpha (float): Alpha parameter for PFSA. Default: 1.0
            do_pcr (bool): Whether to apply PCR. Default: True
            do_pfsa (bool): Whether to apply PFSA. Default: True
        """
        super().__init__()
        self.in_dim = in_dim
        self.target_rank = target_rank
        self.npc = npc
        self.center = center
        self.num_iters = num_iters
        self.do_pcr = do_pcr
        self.do_pfsa = do_pfsa
        
        # Initialize PFSA attention module
        self.attention = PFSA(in_dim, alpha=alpha)
    
    def _compute_pc(
        self, 
        X: torch.Tensor, 
        attention_mask: torch.Tensor
    ) -> list:
        """
        Compute principal components for each sample in the batch.
        
        Args:
            X (torch.Tensor): Input tensor of shape [B, T, F]
            attention_mask (torch.Tensor): Attention mask of shape [B, T]
            
        Returns:
            list: List of principal components for each sample
        """
        principal_components = []
        batch_size, seq_len, feature_dim = X.shape
        
        for sample, mask in zip(X, attention_mask):
            # Get valid sequence length and extract valid tokens
            valid_length = int(mask.sum().item())
            valid_sample = sample[:valid_length, :]
            
            # Determine rank for PCA (minimum of target_rank and valid_length)
            pca_rank = min(self.target_rank, valid_length)
            
            # Perform PCA to get principal components
            _, _, V = torch.pca_lowrank(
                valid_sample, 
                q=pca_rank, 
                center=self.center, 
                niter=self.num_iters
            )
            
            # Extract the first 'npc' principal components
            pc = V.transpose(0, 1)[:self.npc, :]  # Shape: [npc, F]
            principal_components.append(pc)
        
        return principal_components

    def _remove_pc(
        self, 
        X: torch.Tensor, 
        principal_components: list
    ) -> torch.Tensor:
        """
        Remove principal components from input features.
        
        Args:
            X (torch.Tensor): Input tensor of shape [B, T, F]
            principal_components (list): List of principal components
            
        Returns:
            torch.Tensor: Feature tensor with principal components removed
        """
        batch_size, seq_len, feature_dim = X.shape
        output_list = []
        
        for sample, pc in zip(X, principal_components):
            # Remove principal components using projection
            # v = x - x @ pc^T @ pc
            projected_features = sample - sample @ pc.transpose(0, 1) @ pc
            output_list.append(projected_features[None, ...])
        
        output = torch.vstack(output_list)
        return output

    def forward(
        self, 
        input_values: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None, 
        *args, 
        **kwargs
    ) -> torch.Tensor:
        """
        Forward pass: Apply PCR followed by PFSA.
        
        Args:
            input_values (torch.Tensor): Input tensor of shape [B, T, F]
            attention_mask (torch.Tensor, optional): Attention mask of shape [B, T]
            
        Returns:
            torch.Tensor: Processed tensor of shape [B, T, F]
        """
        x = input_values
        
        # Step 1: Apply Principal Component Removal (PCR)
        if self.do_pcr:
            principal_components = self._compute_pc(x, attention_mask)
            x = self._remove_pc(x, principal_components)
        
        # Step 2: Apply Principal Feature Selection Attention (PFSA)
        if self.do_pfsa:
            x = self.attention(x, attention_mask)
        
        return x


class StatisticsPooling(torch.nn.Module):
    """
    Statistics Pooling Module
    
    Computes mean and/or standard deviation statistics across the time dimension,
    with support for attention masking and Gaussian noise augmentation.
    """
    
    def __init__(self, return_mean: bool = True, return_std: bool = True):
        """
        Initialize Statistics Pooling module.
        
        Args:
            return_mean (bool): Whether to compute and return mean statistics. Default: True
            return_std (bool): Whether to compute and return std statistics. Default: True
            
        Raises:
            ValueError: If both return_mean and return_std are False
        """
        super().__init__()
        
        self.eps = 1e-5  # Small value for numerical stability and Gaussian noise
        self.return_mean = return_mean
        self.return_std = return_std
        
        if not (self.return_mean or self.return_std):
            raise ValueError(
                "Both mean and std statistics are disabled. "
                "Please enable at least one statistic (mean and/or std) for pooling."
            )

    def forward(
        self, 
        input_values: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute statistics pooling over the time dimension.
        
        Args:
            input_values (torch.Tensor): Input tensor of shape [B, T, F]
            attention_mask (torch.Tensor, optional): Attention mask of shape [B, T]
            
        Returns:
            torch.Tensor: Pooled statistics tensor of shape [B, 1, F] or [B, 1, 2*F]
        """
        x = input_values
        
        if attention_mask is None:
            # Simple case: no masking, compute statistics over full sequence
            if self.return_mean:
                mean = x.mean(dim=1)
            if self.return_std:
                std = x.std(dim=1)
        else:
            # Complex case: apply attention masking
            mean_list = []
            std_list = []
            
            for sample_idx in range(x.shape[0]):
                # Calculate actual sequence length from attention mask
                sequence_lengths = torch.sum(attention_mask, dim=1)
                relative_lengths = sequence_lengths / torch.max(sequence_lengths)
                actual_size = torch.round(
                    relative_lengths[sample_idx] * x.shape[1]
                ).int()
                
                # Compute statistics only on valid (non-padded) tokens
                valid_tokens = x[sample_idx, 0:actual_size, ...]
                
                if self.return_mean:
                    sample_mean = torch.mean(valid_tokens, dim=0)
                    mean_list.append(sample_mean)
                
                if self.return_std:
                    sample_std = torch.std(valid_tokens, dim=0)
                    std_list.append(sample_std)
            
            # Stack individual sample statistics
            if self.return_mean:
                mean = torch.stack(mean_list)
            if self.return_std:
                std = torch.stack(std_list)

        # Add Gaussian noise to mean for regularization
        if self.return_mean:
            gaussian_noise = self._get_gauss_noise(mean.size(), device=mean.device)
            mean += gaussian_noise
        
        # Add epsilon to std for numerical stability
        if self.return_std:
            std = std + self.eps

        # Concatenate mean and std statistics
        if self.return_mean and self.return_std:
            pooled_stats = torch.cat((mean, std), dim=1)
        elif self.return_mean:
            pooled_stats = mean
        elif self.return_std:
            pooled_stats = std
        
        # Add time dimension back
        pooled_stats = pooled_stats.unsqueeze(1)
        
        return pooled_stats

    def _get_gauss_noise(
        self, 
        shape_of_tensor: torch.Size, 
        device: str = "cpu"
    ) -> torch.Tensor:
        """
        Generate scaled Gaussian noise for regularization.
        
        Args:
            shape_of_tensor (torch.Size): Shape of the tensor for noise generation
            device (str): Device to create the noise tensor on. Default: "cpu"
            
        Returns:
            torch.Tensor: Scaled Gaussian noise tensor
        """
        # Generate random Gaussian noise
        gaussian_noise = torch.randn(shape_of_tensor, device=device)
        
        # Normalize noise to [0, 1] range
        gaussian_noise -= torch.min(gaussian_noise)
        gaussian_noise /= torch.max(gaussian_noise)
        
        # Scale noise to desired range: eps * ((1 - 9) * noise + 9) = eps * (9 - 8 * noise)
        gaussian_noise = self.eps * ((1 - 9) * gaussian_noise + 9)
        
        return gaussian_noise
