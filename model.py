"""
Transformer Model Architecture (Vaswani et al., 2017).

Complete implementation of the original Transformer for Arabic→English NMT.

Build order (matches paper Section 3.1):
1. PositionalEncoding - sinusoidal (no learned params)
2. MultiHeadAttention - scaled dot-product: softmax(QKᵀ/√dk)V
3. PositionwiseFeedForward - Linear→ReLU→Linear
4. EncoderLayer - self-attn → FFN, 2× (residual + LayerNorm)
5. DecoderLayer - masked self-attn → cross-attn → FFN, 3× (residual + LayerNorm)
6. Encoder - stacked EncoderLayer + embedding + pos_enc
7. Decoder - stacked DecoderLayer + embedding + pos_enc
8. Transformer - enc + dec + final linear → logits

Design rationale:
- Modular: each component testable in isolation
- Well-commented: teaches through implementation
- No abstraction leaks: attention, masking, pos encoding explicit
"""

import math
import copy
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import config


# ========== Positional Encoding ==========


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding (Vaswani et al., Section 3.5).

    Why sinusoidal:
    - Allows model to attend to relative positions
    - Can extrapolate to longer sequences than seen in training
    - No learned parameters (more data-efficient)

    Paper: "We use sine and cosine functions of different frequencies:
    PE(pos, 2i) = sin(pos/10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos/10000^(2i/d_model))"

    Where pos = position, i = dimension index.
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        """
        Args:
            d_model: Embedding dimension (must match Transformer d_model)
            max_len: Maximum sequence length to pre-compute
            dropout: Dropout probability
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Pre-compute positional encodings for all positions up to max_len
        # Shape: [max_len, 1] (broadcasts to [max_len, d_model])
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # Compute frequency bands: 1/10000^(2i/d_model) for i in [0, d_model/2)
        # div_term: [d_model // 2] = [e^(2i * -log(10000) / d_model)] for even i
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )

        # Apply sine to even indices: PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        pe[:, 0::2] = torch.sin(position * div_term)

        # Apply cosine to odd indices: PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add batch dimension: [max_len, d_model] → [1, max_len, d_model]
        pe = pe.unsqueeze(0)

        # Register as buffer (not a parameter, but persisted in state_dict)
        # Why buffer: more efficient than computing on every forward pass
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input embeddings.

        Args:
            x: Input tensor, shape [batch, seq_len, d_model]

        Returns:
            Tensor with positional encoding added, same shape
        """
        # x shape: [batch, seq_len, d_model]
        # pe shape: [1, max_len, d_model]
        # Slice pe to match seq_len, add to x
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


# ========== Multi-Head Attention ==========


class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention (Vaswani et al., Section 3.2).

    Paper equation:
    Attention(Q, K, V) = softmax(QKᵀ / √dk)V

    Where:
    - Q, K, V: queries, keys, values (projected from input)
    - dk: key dimension (d_model / num_heads)

    Why multi-head:
    - Allows model to attend to different representation subspaces
    - Some heads can focus on syntax, others on semantics
    - Multiple attention patterns in parallel
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
    ):
        """
        Args:
            d_model: Model dimension (256 in our config)
            num_heads: Number of attention heads (8)
            dropout: Dropout probability
        """
        super().__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.dk = d_model // num_heads  # Dimension per head: 256/8 = 32

        # Linear projections for Q, K, V (single large matrix split into heads)
        # Why one matrix per head: more efficient than individual matrices
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # Output projection (combine heads back to d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(p=dropout)

        # Cache for attention weights (for visualization)
        self.last_attention_weights: Optional[torch.Tensor] = None

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for multi-head attention.

        Args:
            query: Shape [batch, seq_len_q, d_model]
            key: Shape [batch, seq_len_k, d_model]
            value: Shape [batch, seq_len_v, d_model]
            mask: Optional attention mask [batch, 1, seq_len_k] or [batch, seq_len_q, seq_len_k]

        Returns:
            Output: [batch, seq_len_q, d_model]
        """
        batch_size = query.size(0)

        # Step 1: Project and split into heads
        # [batch, seq, d_model] → [batch, seq, d_model] via linear
        # Then reshape: [batch, seq, d_model] → [batch, num_heads, seq, dk]
        Q = (
            self.W_q(query)
            .view(batch_size, -1, self.num_heads, self.dk)
            .transpose(1, 2)
        )
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.dk).transpose(1, 2)
        V = (
            self.W_v(value)
            .view(batch_size, -1, self.num_heads, self.dk)
            .transpose(1, 2)
        )

        # Step 2: Scaled dot-product attention
        # scores = Q @ K^T / √dk
        # Shape: [batch, num_heads, seq_len_q, seq_len_k]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.dk)

        # Step 3: Apply mask if provided
        # Typical masks:
        # - Padding mask: [batch, 1, 1, seq_len_k] - same mask for all query positions
        # - Look-ahead mask: [batch, 1, seq_len_q, seq_len_k] - different per query
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        # Step 4: Softmax to get attention weights
        # Why softmax: outputs sum to 1 (weighted sum of values)
        attention_weights = F.softmax(scores, dim=-1)
        self.last_attention_weights = attention_weights  # Cache for visualization

        # Apply dropout to attention weights (during training only)
        attention_weights = self.dropout(attention_weights)

        # Step 5: Apply attention to values
        # output = attention_weights @ V
        # Shape: [batch, num_heads, seq_len_q, dk]
        output = torch.matmul(attention_weights, V)

        # Step 6: Concatenate heads and project
        # [batch, num_heads, seq_len_q, dk] → [batch, seq_len_q, d_model]
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_o(output)

        return output


# ========== Positionwise Feed-Forward ==========


class PositionwiseFeedForward(nn.Module):
    """
    Position-wise feed-forward network (Vaswani et al., Section 3.3).

    Paper: "The feed-forward networks consist of two linear transformations
    with a ReLU activation in between."

    FFN(x) = Linear(ReLU(Linear(x)))

    Why separate per position:
    - Adds non-linearity between attention layers
    - Increases model capacity
    - Two linear layers = one hidden layer (single hidden layer neural network)

    Hidden dimension is typically 4× d_model (512 → 2048 in paper).
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        """
        Args:
            d_model: Input/output dimension
            d_ff: Hidden dimension (typically 4× d_model)
            dropout: Dropout probability
        """
        super().__init__()

        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: [batch, seq_len, d_model]

        Returns:
            [batch, seq_len, d_model]
        """
        # ReLU activation between two linear layers
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


# ========== Encoder Layer ==========


class EncoderLayer(nn.Module):
    """
    Single Transformer encoder layer (Vaswani et al., Figure 1 left).

    Each encoder layer consists of:
    1. Multi-head self-attention (with padding mask)
    2. Positionwise feed-forward
    3. Two residual connections + layer normalization

    Paper Section 5.4: "We employ a residual connection around each
    of the two sub-layers, followed by layer normalization."
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)

        # Two layer norms: after attention and after FFN
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        x: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through encoder layer.

        Args:
            x: [batch, src_len, d_model]
            src_mask: Source padding mask

        Returns:
            [batch, src_len, d_model]
        """
        # Step 1: Multi-head self-attention with residual
        # Self-attention: Q = K = V = x (attends to own representation)
        attn_output = self.self_attn(x, x, x, src_mask)
        x = self.norm1(x + self.dropout(attn_output))

        # Step 2: Feed-forward with residual
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))

        return x


# ========== Decoder Layer ==========


class DecoderLayer(nn.Module):
    """
    Single Transformer decoder layer (Vaswani et al., Figure 1 right).

    Each decoder layer consists of:
    1. Masked multi-head self-attention (prevents attending to future positions)
    2. Multi-head cross-attention (attends to encoder output)
    3. Positionwise feed-forward
    4. Three residual connections + layer normalization

    The masked self-attention ensures teacher forcing during training:
    position i can only attend to positions 0 to i-1.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Masked self-attention (for target sequence)
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)

        # Cross-attention (for encoder-decoder attention)
        # Queries from decoder, Keys/Values from encoder output
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)

        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)

        # Three layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(p=dropout)

        # Cache for cross-attention weights (for visualization)
        self.last_cross_attention_weights: Optional[torch.Tensor] = None

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        src_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through decoder layer.

        Args:
            x: [batch, tgt_len, d_model] (target embeddings)
            encoder_output: [batch, src_len, d_model] (encoder output)
            tgt_mask: Target padding mask
            src_mask: Source padding mask

        Returns:
            [batch, tgt_len, d_model]
        """
        # Step 1: Masked self-attention (with look-ahead mask)
        # Mask ensures position i cannot see positions > i
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))

        # Step 2: Cross-attention (encoder-decoder attention)
        # Q from decoder, K/V from encoder
        attn_output = self.cross_attn(x, encoder_output, encoder_output, src_mask)

        # Cache for visualization
        self.last_cross_attention_weights = self.cross_attn.last_attention_weights

        x = self.norm2(x + self.dropout(attn_output))

        # Step 3: Feed-forward with residual
        ffn_output = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_output))

        return x


# ========== Encoder ==========


class Encoder(nn.Module):
    """
    Transformer Encoder (Vaswani et al., Section 3.1).

    Stacks N identical encoder layers (N=3 in our config).

    Each layer processes the representation, allowing the model to:
    - Build hierarchical representations of the input
    - Attend to different aspects of the source sequence
    """

    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        vocab_size: int,
        max_len: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.num_layers = num_layers
        self.d_model = d_model

        # Embedding layer (learned)
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=config.PAD_ID)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)

        # Stack of encoder layers
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )

        # Final layer norm (Vaswani et al., Section 5.4)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through encoder.

        Args:
            src: Source token IDs, shape [batch, src_len]
            src_mask: Source padding mask

        Returns:
            Encoder output, shape [batch, src_len, d_model]
        """
        # Embed source tokens and add positional encoding
        # [batch, src_len] → [batch, src_len, d_model]
        x = self.embedding(src) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)

        # Pass through each encoder layer
        for layer in self.layers:
            x = layer(x, src_mask)

        # Final layer norm
        x = self.norm(x)

        return x

    def get_attention_weights(self, layer_idx: int = -1) -> Optional[torch.Tensor]:
        """
        Get attention weights from a specific layer for visualization.

        Args:
            layer_idx: Layer index (default: last layer)

        Returns:
            Attention weights [batch, num_heads, seq_len, seq_len]
        """
        if abs(layer_idx) >= self.num_layers:
            return None
        return self.layers[layer_idx].self_attn.last_attention_weights


# ========== Decoder ==========


class Decoder(nn.Module):
    """
    Transformer Decoder (Vaswani et al., Section 3.1).

    Stacks N identical decoder layers (N=3 in our config).

    Cross-attention allows each decoder position to attend
    to all encoder positions (source information).
    """

    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        vocab_size: int,
        max_len: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.num_layers = num_layers
        self.d_model = d_model

        # Embedding layer (learned)
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=config.PAD_ID)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)

        # Stack of decoder layers
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )

        # Final layer norm
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        tgt: torch.Tensor,
        encoder_output: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        src_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through decoder.

        Args:
            tgt: Target token IDs (with BOS), shape [batch, tgt_len]
            encoder_output: Encoder output, shape [batch, src_len, d_model]
            tgt_mask: Target padding/look-ahead mask
            src_mask: Source padding mask

        Returns:
            Decoder output, shape [batch, tgt_len, d_model]
        """
        # Embed target tokens and add positional encoding
        x = self.embedding(tgt) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)

        # Pass through each decoder layer
        for layer in self.layers:
            x = layer(x, encoder_output, tgt_mask, src_mask)

        # Final layer norm
        x = self.norm(x)

        return x

    def get_attention_weights(
        self, layer_idx: int = -1, attention_type: str = "cross"
    ) -> Optional[torch.Tensor]:
        """
        Get attention weights from a specific layer for visualization.

        Args:
            layer_idx: Layer index (default: last layer)
            attention_type: 'self' or 'cross'

        Returns:
            Attention weights [batch, num_heads, tgt_len, src_len or tgt_len]
        """
        if abs(layer_idx) >= self.num_layers:
            return None

        layer = self.layers[layer_idx]
        if attention_type == "cross":
            return layer.last_cross_attention_weights
        else:
            return layer.self_attn.last_attention_weights


# ========== Transformer ==========


class Transformer(nn.Module):
    """
    Complete Transformer model (Vaswani et al., Figure 1).

    Architecture:
    - Encoder: processes source sequence
    - Decoder: generates target sequence
    - Final linear: projects to vocabulary logits

    Forward pass:
    1. Encode source with encoder
    2. Decode with cross-attention to encoder
    3. Project to vocabulary for next-token prediction
    """

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 3,
        d_ff: int = 512,
        max_len: int = 50,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.d_model = d_model
        self.tgt_vocab_size = tgt_vocab_size

        # Encoder
        self.encoder = Encoder(
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            vocab_size=src_vocab_size,
            max_len=max_len,
            dropout=dropout,
        )

        # Decoder
        self.decoder = Decoder(
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            vocab_size=tgt_vocab_size,
            max_len=max_len,
            dropout=dropout,
        )

        # Final linear layer (project to vocabulary)
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)

        # Initialize weights (standard Transformer initialization)
        self._init_weights()

    def _init_weights(self):
        """
        Initialize weights with Xavier uniform.

        Paper: "We use the uniform distribution for initialization"
        Per Vi: "All model parameters are initialized with uniform distribution"
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def generate_padding_mask(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generate padding mask for input tensor.

        Where x[i] == PAD_ID, mask[i] = 0 (blocked)
        Elsewhere mask[i] = 1 (allowed)

        Args:
            x: Input tensor of token IDs, shape [batch, seq_len]

        Returns:
            Mask tensor, shape [batch, 1, 1, seq_len]
        """
        # Padding positions: where x == PAD_ID
        padding_mask = (x != config.PAD_ID).unsqueeze(1).unsqueeze(2)
        return padding_mask

    def generate_look_ahead_mask(
        self, seq_len: int, device: torch.device
    ) -> torch.Tensor:
        """
        Generate look-ahead (causal) mask for decoder.

        Ensures position i can only attend to positions 0 to i-1.
        This is CRITICAL for training: prevents "cheating" by looking at target.

        Example for seq_len=5:
        [[1, 0, 0, 0, 0],   # position 0: attend to none before
         [1, 1, 0, 0, 0],   # position 1: attend to 0
         [1, 1, 1, 0, 0],   # position 2: attend to 0,1
         [1, 1, 1, 1, 0],   # position 3: attend to 0,1,2
         [1, 1, 1, 1, 1]]   # position 4: attend to all before

        Args:
            seq_len: Sequence length
            device: Device for tensor

        Returns:
            Look-ahead mask, shape [1, 1, seq_len, seq_len]
        """
        # Create upper triangular matrix (1s on and below diagonal)
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device), diagonal=1
        ).bool()

        # Invert so 1s become allowed positions
        mask = ~mask

        # Add dimensions for broadcasting: [seq_len, seq_len] → [1, 1, seq_len, seq_len]
        mask = mask.unsqueeze(0).unsqueeze(0)

        return mask

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass through Transformer.

        Args:
            src: Source token IDs, [batch, src_len]
            tgt: Target token IDs (with BOS), [batch, tgt_len]

        Returns:
            Logits, [batch, tgt_len, tgt_vocab_size]
        """
        # Generate masks
        src_padding_mask = self.generate_padding_mask(src)

        # For decoder: both padding mask and look-ahead mask
        tgt_padding_mask = self.generate_padding_mask(tgt)
        look_ahead_mask = self.generate_look_ahead_mask(tgt.size(1), tgt.device)

        # Combine padding and look-ahead: both must be satisfied
        tgt_mask = tgt_padding_mask & look_ahead_mask

        # Encode source
        encoder_output = self.encoder(src, src_padding_mask)

        # Decode
        decoder_output = self.decoder(
            tgt,
            encoder_output,
            tgt_mask=tgt_mask,
            src_mask=src_padding_mask,
        )

        # Project to vocabulary
        logits = self.fc_out(decoder_output)

        return logits

    def get_cross_attention_weights(
        self, layer_idx: int = -1
    ) -> Optional[torch.Tensor]:
        """Get cross-attention weights for visualization."""
        return self.decoder.get_attention_weights(layer_idx, attention_type="cross")


# ========== Smoke Test ==========

if __name__ == "__main__":
    # Simple smoke test to verify shapes
    print("Running smoke test...")

    # Create model
    model = Transformer(
        src_vocab_size=8000,
        tgt_vocab_size=8000,
        d_model=256,
        num_heads=8,
        num_layers=3,
        d_ff=512,
    )
    model.eval()

    # Dummy input
    batch_size = 2
    src_len = 10
    tgt_len = 12

    src = torch.randint(4, 8000, (batch_size, src_len))
    tgt = torch.randint(4, 8000, (batch_size, tgt_len))

    # Forward pass
    with torch.no_grad():
        logits = model(src, tgt)

    print(f"src shape: {src.shape}")
    print(f"tgt shape: {tgt.shape}")
    print(f"logits shape: {logits.shape}")
    print(f"Expected: ({batch_size}, {tgt_len}, 8000)")

    # Verify
    assert logits.shape == (batch_size, tgt_len, 8000), "Output shape mismatch!"
    print("✓ Smoke test passed!")

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
