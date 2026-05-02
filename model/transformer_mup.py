"""
µP (Maximal Update Parameterization) GPT Model
Implements µP scaling rules for zero-shot hyperparameter transfer across model widths.
Based on: Yang et al. (2022), "Tensor Programs V"
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

try:
    import mup
    from mup import MuReadout, set_base_shapes, MuAdamW
    MUP_AVAILABLE = True
except ImportError:
    print("WARNING: mup package not installed. Install with: pip install mup")
    MUP_AVAILABLE = False


@dataclass
class MuPGPTConfig:
    """Configuration for µP GPT model."""
    vocab_size: int = 4096
    block_size: int = 1024
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    d_ff: int = 1536
    dropout: float = 0.0
    bias: bool = False


class MuPCausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention with µP scaling.
    Key difference: attention scores scaled by 1/d instead of 1/sqrt(d).
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout

        # µP: Using 1/d scaling instead of 1/sqrt(d)
        self.attn_scale = 1.0 / self.head_dim  # NOT 1/sqrt(d)

        # Causal mask
        self.register_buffer("causal_mask",
            torch.tril(torch.ones(config.block_size, config.block_size))
            .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()

        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # µP attention: scaling by 1/d (not 1/sqrt(d))
        att = (q @ k.transpose(-2, -1)) * self.attn_scale
        att = att.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class MuPMLP(nn.Module):
    """Feed-forward network (same structure, µP handles init/lr externally)."""

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, config.d_ff, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(config.d_ff, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class MuPBlock(nn.Module):
    """Transformer block with pre-LayerNorm for µP model."""

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = MuPCausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MuPMLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class MuPGPT(nn.Module):
    """
    GPT Language Model with µP (Maximal Update Parameterization).

    Key differences from standard GPT:
    1. Output layer uses MuReadout (from mup package)
    2. Attention scaling: 1/d instead of 1/sqrt(d)
    3. base_shapes set from smallest model for HP transfer
    4. Uses MuAdamW optimizer with per-layer LR multipliers
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([MuPBlock(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd, bias=config.bias),
        ))

        # µP: Using MuReadout for the output projection
        if MUP_AVAILABLE:
            self.lm_head = MuReadout(config.n_embd, config.vocab_size, bias=False)
        else:
            self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Initialize weights
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        B, T = idx.size()
        assert T <= self.config.block_size

        pos = torch.arange(0, T, dtype=torch.long, device=device)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    def count_parameters(self, non_embedding=True):
        total = sum(p.numel() for p in self.parameters())
        if non_embedding:
            emb_params = self.transformer.wte.weight.numel() + self.transformer.wpe.weight.numel()
            return total - emb_params
        return total

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = -float('Inf')

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

    @classmethod
    def from_config_dict(cls, config_dict, vocab_size=4096, block_size=1024, dropout=0.0):
        config = MuPGPTConfig(
            vocab_size=vocab_size,
            block_size=block_size,
            n_layer=config_dict['n_layer'],
            n_head=config_dict['n_head'],
            n_embd=config_dict['n_embd'],
            d_ff=config_dict['d_ff'],
            dropout=dropout,
        )
        return cls(config)


def setup_mup(model, base_config_dict, vocab_size=4096, block_size=1024, save_path=None):
    """
    Set up µP base shapes for a model.

    Args:
        model: The target MuPGPT model
        base_config_dict: Config dict for the smallest (base) model
        vocab_size: Vocabulary size
        block_size: Context window size
        save_path: Optional path to save base shapes

    Returns:
        model: Model with µP base shapes set
    """
    if not MUP_AVAILABLE:
        raise ImportError("mup package required. Install with: pip install mup")

    # Create base model
    base_model = MuPGPT.from_config_dict(
        base_config_dict, vocab_size=vocab_size, block_size=block_size
    )

    # Create delta model (slightly wider, for shape inference)
    # Ensure delta n_embd is divisible by n_head to keep attention structure
    delta_config = base_config_dict.copy()
    base_n_head = base_config_dict['n_head']
    delta_config['n_embd'] = base_config_dict['n_embd'] + base_n_head  # stays divisible by n_head
    delta_config['d_ff'] = base_config_dict['d_ff'] + 4
    delta_model = MuPGPT.from_config_dict(
        delta_config, vocab_size=vocab_size, block_size=block_size
    )

    # Set base shapes
    set_base_shapes(model, base_model, delta=delta_model)

    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print(f"Saved µP base shapes to {save_path}")

    return model


def get_mup_optimizer(model, lr, weight_decay=0.1, betas=(0.9, 0.95)):
    """
    Get MuAdamW optimizer with µP learning rate scaling.

    Args:
        model: MuPGPT model with base shapes set
        lr: Base learning rate (tuned on smallest model)
        weight_decay: Weight decay
        betas: Adam betas

    Returns:
        MuAdamW optimizer
    """
    if not MUP_AVAILABLE:
        raise ImportError("mup package required")

    return MuAdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=betas,
    )
