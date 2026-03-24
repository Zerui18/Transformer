# Transformer Lab

A from-scratch PyTorch playground for transformer architectures — attention mechanisms, seq2seq models, tokenization, datasets, and automated experiment sweeps, all on a single GPU.

## Project Structure

```
├── modules/                    # Core neural network building blocks
│   ├── embedding.py            # Sinusoidal + learnable token embeddings
│   ├── transformer.py          # Encoder/decoder blocks, FFN, LM head
│   ├── whisper.py              # CNN-based audio feature encoder
│   ├── softmax1.py             # Experimental softmax variant (sum + 1 normalizer)
│   └── attention/              # Pluggable attention mechanisms
│       ├── base.py             # Abstract base classes (self & cross attention)
│       ├── vanilla.py          # Standard scaled dot-product
│       ├── roformer_attn.py    # Rotary position embeddings (RoPE)
│       ├── multi_query.py      # Multi-query attention
│       ├── stock.py            # PyTorch's F.scaled_dot_product_attention
│       ├── flash.py            # Flash attention backend
│       ├── meme.py             # Memory-efficient attention backend
│       ├── softmax1.py         # Attention with softmax1
│       ├── average.py          # Uniform averaging (no softmax)
│       └── xformer.py          # xFormers integration
│
├── models/                     # Full model architectures (Lightning modules)
│   ├── transformer.py          # Encoder-decoder Transformer for translation
│   ├── whisper.py              # Whisper-style speech-to-text
│   └── hf_encoder_decoder.py   # HuggingFace EncoderDecoder wrapper
│
├── datasets/                   # Dataset implementations
│   ├── base.py                 # Abstract base
│   ├── translate.py            # SentencePiece tokenized parallel text
│   ├── translate_spacy.py      # Spacy tokenized parallel text
│   ├── translate_spacy_multi30k.py  # TorchText Multi30K with Spacy
│   ├── atis.py                 # ATIS speech-to-text (mel spectrograms)
│   └── textgen.py              # Text generation dataset
│
├── toknizers/                  # Tokenizer abstractions
│   ├── tokenizer.py            # Abstract base
│   └── SPTokenizer.py          # SentencePiece wrapper
│
├── metrics/
│   └── bleu.py                 # BLEU-4 via NLTK
│
├── exp/                        # Experiment management system
│   ├── experiment.py           # Experiment runner & config loader
│   ├── manager.py              # Queue-based experiment scheduler
│   └── server.py               # Flask REST API for remote control
│
├── configs/                    # YAML config sets (model + dls + trainer)
│   ├── de-en-v1-sp-multi30k/   # DE→EN translation, SentencePiece
│   ├── de-en-v1-spacy-mutli30k/
│   ├── de-en-v1-spacymy-mutli30k/
│   ├── de-en-hf-spacy-mutli30k/
│   └── atis-v1/                # ATIS speech-to-text
│
├── efficient_training/         # DeepSpeed ZeRO stage 2/3 configs
│
├── run_single_exp.py           # Run one experiment from CLI
├── test_multi30k_v1.py         # Sweep: blocks × vocab × attention type
├── test_multi30k_v1_misc.py    # Sweep: blocks × heads × dim × weight tying
├── test_multi30k_v1_stability.py  # Repeated runs for variance
├── test_atis_v1.py             # Sweep: blocks × CNN layers × vocab size
├── translate.py                # Interactive greedy translation
├── translate_beam.py           # Interactive beam search (curses TUI)
└── whisper_preprocess.py       # Mel spectrogram extraction & normalization
```

## Quick Start

### Run a single experiment

```bash
python run_single_exp.py \
  --exp-name my_experiment \
  --exp-save-path experiments/ \
  --model-config-path configs/de-en-v1-sp-multi30k/model.yaml \
  --dls-config-path configs/de-en-v1-sp-multi30k/dls.yaml \
  --trainer-config-path configs/de-en-v1-sp-multi30k/trainer.yaml
```

### Run an experiment sweep

```python
from exp.manager import ExperimentManager, ExperimentConfig

manager = ExperimentManager('experiments/my_sweep')

for n_blocks in [4, 5, 6]:
    config = ExperimentConfig.from_config_files(
        'configs/de-en-v1-sp-multi30k/model.yaml',
        'configs/de-en-v1-sp-multi30k/dls.yaml',
        'configs/de-en-v1-sp-multi30k/trainer.yaml',
    )
    config.model_config['init_args']['n_blocks'] = n_blocks
    manager.create_and_append_experiment(f'blocks_{n_blocks}', config)
```

Experiments run sequentially (one GPU), with automatic progression through the queue.

### Interactive translation

```bash
# Greedy decoding
python translate.py --model transformer --model-path <ckpt> --tokenizer-path <sp_model>

# Beam search (curses TUI)
python translate_beam.py --model transformer --model-path <ckpt> --tokenizer-path <sp_model>
```

## Configuration

Each experiment is defined by three YAML files:

**model.yaml** — architecture, optimizer, attention type, weight tying
```yaml
class: Transformer
init_args:
  max_len: 512
  src_vocab_size: 15000
  tgt_vocab_size: 15000
  n_blocks: 6
  n_heads: 8
  emb_dim: 512
  dropout: 0.1
  bias: false
  weight_tying: 3-way    # 3-way | 2-way | null
  attention_type: roformer_attn  # vanilla | roformer_attn | multi_query | stock | flash | meme | ...
  optimizer: AdamW
  learning_rate: 0.0005
tokenizer:
  class: SPTokenizer
  init_args:
    sp_model_path: data/multi30k/m_en_de.model
```

**dls.yaml** — dataset class, file paths, batch sizes

**trainer.yaml** — Lightning Trainer args (epochs, gradient clipping, logging frequency)

## Attention Mechanisms

New attention variants are added by creating a module in `modules/attention/` that exposes `MultiHeadSelfAttention` and `MultiHeadCrossAttention` classes inheriting from the base classes. The mechanism is selected at runtime via the `attention_type` string in model config.

## Experiment Manager

The `ExperimentManager` maintains four queues (queued, running, completed, stopped, failed) and runs experiments one at a time. In multi-process mode, each experiment runs in a child process with shared state via `multiprocessing.Value`. A background scheduler polls for completion/failure and auto-advances the queue.

The Flask server in `exp/server.py` exposes REST endpoints for queue inspection and control.
