from dataclasses import dataclass

class Decodable(object):
    def __init__(self, data):
        for name, value in data.items():
            setattr(self, name, self._wrap(value))

    def _wrap(self, value):
        if isinstance(value, (tuple, list, set, frozenset)): 
            return type(value)([self._wrap(v) for v in value])
        else:
            return Decodable(value) if isinstance(value, dict) else value

@dataclass(frozen=True)
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

@dataclass(frozen=True)
class TrainingConfig:
    learning_rate: float 
    autotune_learning_rate: bool 
    compile_model: bool
    shuffle: bool
    steps: int 
    batch_size: int
    autotune_batch_size: bool
    gradient_accum_steps: int 
    sp_model: str
    train_src_file: str
    train_dst_file: str
    val_src_file: str
    val_dst_file: str