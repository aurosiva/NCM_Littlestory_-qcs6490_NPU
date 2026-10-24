"""Minimal NCM (Next-Concept Modeling, arXiv:2610.14162) config, sized for the
Radxa Q6A (QCS6490). The paper's TinyStories setup uses a 36M tokenizer + 89M
generator; this one is ~5M + ~16M so it runs comfortably on the board's CPU.
"""
from dataclasses import dataclass, field


@dataclass
class TokCfg:  # stage 1: multiscale residual VQ-VAE
    vocab: int = 4096          # text BPE vocab (paper: 8192)
    L: int = 128               # tokens per sample (same as paper)
    # level lengths l_1 < ... < l_K = L (paper uses K=12 for TinyStories)
    levels: list = field(default_factory=lambda: [1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128])
    codebook: int = 1024       # codes per level (paper: 4096)
    d: int = 256               # latent width (paper: 384)
    enc_layers: int = 0        # paper uses 0 encoder blocks
    dec_layers: int = 2        # paper: 1
    heads: int = 4
    dropout: float = 0.1
    ema_decay: float = 0.95
    commit_beta: float = 0.25
    latent_w: float = 1.0   # VAR uses beta*commit + 1.0*latent
    nested_dropout: float = 0.5  # prob. of truncating at a random level
    # v2 options (from the official HazyResearch code); defaults = baseline behaviour
    l1_learned: bool = False       # learned strided-conv down/up for level 1 instead of avg/linear
    per_level_vq_loss: bool = False  # commit + latent loss after every level (averaged), not only at the end

    @property
    def K(self):
        return len(self.levels)


@dataclass
class GenCfg:  # stage 2: NCM transformer
    width: int = 384           # paper: 768
    layers: int = 6            # paper: 12
    heads: int = 6
    dropout: float = 0.1
    alpha: float = 0.5         # level loss weight w_k ∝ l_k^-alpha
    corrupt_hi: float = 0.75   # code corruption rate at level 1 ...
    corrupt_lo: float = 0.25   # ... decreasing linearly to level K (mean 0.5, as in paper)
    tied_heads: bool = False   # v2: score codes by distance to the frozen codebook (official code)
    rope: bool = False         # v2: RoPE (positions restart at 0 in every level) instead of a learned table
    rope_base: float = 1e5     # official TinyStories value
    qk_norm: bool = False      # v2: RMSNorm on per-head queries and keys


@dataclass
class TrainCfg:
    tok_steps: int = 8000
    tok_batch: int = 256
    tok_lr: float = 5e-4
    gen_steps: int = 20000
    gen_batch: int = 128
    gen_lr: float = 3e-4
    warmup: int = 500
    lr_floor_ratio: float = 0.1
    weight_decay: float = 0.1


# v2 = baseline + three choices taken from the official code (github.com/HazyResearch/ncm)
V2_TOK = dict(levels=[1, 8, 16, 24, 32, 40, 52, 64, 76, 92, 108, 128],
              l1_learned=True, per_level_vq_loss=True)
V2_GEN = dict(tied_heads=True, rope=True, qk_norm=True)
