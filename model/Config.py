from dataclasses import dataclass

@dataclass
class Config():
    n_layers: int = 6
    n_clip_emb: int = 512
    n_heads: int = 8
    d_model: int = 768
    d_ff: int = 3072
    dropout: float = 0.1
    prefix_length: int = 20
    clip_length: int = 10
    dropout: float = 0.1