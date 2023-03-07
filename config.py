from dataclasses import dataclass

@dataclass
class TransformerConfig:
    block_size: int
    vocab_size: int 
    n_blocks: int 
    n_heads: int 
    emb_dim: int 
    dropout: float
    bias: bool 
    weight_tying: bool
    use_grad_ckpt: bool

@dataclass
class TrainingConfig:
    learning_rate: float 
    autotune_learning_rate: bool 
    compile_model: bool
    shuffle: bool
    max_steps: int 
    batch_size: int
    gradient_accum_steps: int 
    sp_model: str
    train_src_file: str
    train_tgt_file: str
    val_src_file: str
    val_tgt_file: str