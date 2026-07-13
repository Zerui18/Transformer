# zlab

A from-scratch, highly customizable PyTorch playground for transformer-like models on DL tasks (and RL in future).

## Stack
- Language: Python 3.12
- Framework: PyTorch + PyTorch Lightning
- Package manager: uv
- Test framework: None (experiment scripts in `test_*.py` are sweep runners, not unit tests)

## Conventions
1. Always clearly define argument and return types in function signatures.
2. Always clearly state the tensor/md-array shapes in both declarations and intermediate computations.
3. Every function/class should have a clear triple-quoted docstring immediately following its declaration (eg, class XXX, def xxx():). It should clearly and concisely describe the purpose and usage of the function and any other additional points to note. It should follow exactly the following format:
```
''' What the function/class does in one sentence.

[for functions]
Args:
	1. {name}: {type} [{expected dtype}, {expected shape}] [pre-conditions] one-sentence description of its purpose
	2. ...
Yields: same format as an arg
Returns: same format as an arg

[for classes]
Properties:
	1. {name}: {type}  [{expected dtype}, {expected shape}] one-sentence description of its purpose
	2. ...

[Additional elaboration if needed & Key points of pay attention to when using the function/class]
'''
```
4. Tensor dtypes are specified as type[+width] (eg, int32/bf16/bool).
5. Tensor shapes are specified as a tuple of symbols (for variable dimensions) and integers (for fixed dimensions), eg, (1, T, C) for fixed batch size of 1, followed by timesteps and channels. Use the following capitalised symbols where applicable:
	- B: batch size
	- T: timesteps
	- C: channels
	- D: embedding dims
	- H: heads
	- E: experts
Each symbol can be further specialised with a lowercased subscript, such as Tq for timesteps of query, Tk for timesteps for keys.
6. Prefer using `einops` for complicated tensor reductions and axes permutations as far as possible and follow the aforementioned conventions to make tensor shapes maximally clear.
7. Intersperse function bodies with comments at critical points to aid understanding, but no need to explain every line.
8. Imports should always be in sequence starting from built-in packages, external packages, project imports.
9. Parameters should always be required unless they have a commonly used value or can be None.
10. Functions should start by pre-conditioning tensor shape/dtype constraints where applicable, raising helpful error messages upon failure.
11. Pytorch code should be device agnostic as far as possible. Initializing new tensors should adaptively adopt the dtype and device within their context of operation.
12. For all modules except for `main`, there shall be no top-level statements besides imports and definitions.

## Architecture

The core architectural goal of this project is to empower simple yet rich compositions while remaining fully customizable. An experiment requires a model (composed of modules), with a dataset (using a tokenizer), and a Lightning trainer. Except for the trainer, all others are defined in this project under their respective directories.

### Registry System (`registry.py`)

A single project-level module that scans the 5 component directories (`modules/`, `models/`, `datasets/`, `tokenizers/`, `metrics/`) at startup and builds a flat lookup table for each. Component directories are plain directories containing `.py` files — they do NOT use `__init__.py`. All class discovery is handled centrally by the registry.

**Registries:**
```python
registry.modules:    dict[str, type[nn.Module]]           # keyed by class name
registry.models:     dict[str, type[LightningModule]]     # keyed by class name
registry.datasets:   dict[str, type[BaseDataset]]         # keyed by class name
registry.tokenizers: dict[str, type[Tokenizer]]           # keyed by class name
registry.metrics:    dict[str, Callable]                   # keyed by function name
```

Each registry also collects companion `@dataclass` config classes found in the same files (eg, `TransformerConfig` alongside `Transformer`). These are accessible from the same dict — configs and their classes share the registry since their names never collide (`Foo` vs `FooConfig`).

**Scanning rules:**
1. Recursively walk the directory for `.py` files, skipping files starting with `_`.
2. Import each module and inspect its members for classes **defined in that file** (ie, `cls.__module__ == mod.__name__`, not re-imports).
3. For each concrete (non-abstract) subclass of the directory's base class, register it by its class name. Also register any `@dataclass` whose name ends with `Config`.
4. **Name collisions raise an error at startup** — every registered class must have a unique name across its registry. This is a hard invariant that prevents silent shadowing.

**Initialisation:** `registry.build()` is called once at the top of every entrypoint (`main.py`, `run_single_exp.py`, `translate.py`, etc.) before any component is used. The experiment system then uses `registry.models['Transformer']` instead of `getattr(models, 'Transformer')`.

**Base classes and abstract classes** (eg, `MultiHeadSelfAttentionBase`, `BaseDataset`, `Tokenizer`) are excluded from registration — only concrete leaf classes are registered. Base classes remain importable via normal Python imports for subclassing.

#### Attention Module Collision

The current codebase has 8 attention files that all export identically-named `MultiHeadSelfAttention` and `MultiHeadCrossAttention`. A flat name-keyed registry cannot hold these. Two options:

**Option A — Rename to unique class names (recommended):**
Each attention variant gets a descriptive unique name. The `attention_type` config string maps to class names via a simple convention:
```
vanilla.py      → VanillaSelfAttention,     VanillaCrossAttention
stock.py        → StockSelfAttention,       StockCrossAttention
flash.py        → FlashSelfAttention,       FlashCrossAttention
roformer_attn.py→ RoFormerSelfAttention,    RoFormerCrossAttention
multi_query.py  → MultiQuerySelfAttention,  MultiQueryCrossAttention
xformer.py      → XFormerSelfAttention,     XFormerCrossAttention
average.py      → AverageSelfAttention,     AverageCrossAttention
softmax1.py     → Softmax1SelfAttention,    Softmax1CrossAttention
meme.py         → MemeSelfAttention,        MemeCrossAttention
```
Transformer blocks resolve attention via registry lookup:
```python
sa_cls = registry.modules[f'{attention_type}SelfAttention']
ca_cls = registry.modules[f'{attention_type}CrossAttention']
```
Config YAML uses the prefix as `attention_type` (eg, `attention_type: Vanilla`). This is clean, explicit, and everything lives in one flat registry.

**Option B — Separate attention sub-registry:**
Keep the current same-name convention. The registry exposes a separate lookup:
```python
registry.attention: dict[str, types.ModuleType]  # keyed by file stem
```
Where `registry.attention['vanilla']` returns the imported module, and callers do `getattr(mod, 'MultiHeadSelfAttention')`. This preserves the polymorphic convention but fragments the registry into two lookup mechanisms.

**Recommendation:** Option A. It's simpler, everything stays in one flat dict, name collisions are caught at startup, and the class names are self-documenting. The overhaul is the right time to make this change.

### Modules (`modules/`)
Subclasses of `torch.nn.Module`. May have further subdirectories for logical grouping (eg, attention modules under `modules/attention/`).

**Constructor convention:** Expose all configurable hparams as explicit keyword arguments, followed by packed hparams for nested modules as `dict[str, Any]`. Hparams should NOT be packed into dataclasses for modules. Nested hparam dicts may include a `cls` key to specify the class name, looked up from `registry.modules`, with remaining keys passed to the constructor.

Example:
```python
def __init__(self,
             hparam1: type1,
             hparam2: type2 = default2,
             hparam3: type3 | None = None,
             nested_hparams4: dict[str, Any] = {}):
```

**Attention modules (`modules/attention/`):** All attention modules must inherit from the abstract base classes in `modules/attention/base.py` (`SelfAttentionBase`, `CrossAttentionBase`) and implement the `_forward()` method. The base class `forward()` handles output wrapping (tuple return, optional attention weights).

All attention modules must satisfy a consistent interface:
- Self-attention: `forward(x, tok_mask) -> tuple[Tensor, ...]`
- Cross-attention: `forward(x_q, x_kv, q_tok_mask, kv_tok_mask) -> tuple[Tensor, ...]`

The return must always be a tuple where `[0]` is the output tensor (callers index with `[0]`). Optional `[1]` is attention weights.

### Models (`models/`)
Subclasses of `lightning.LightningModule`. May have further subdirectories for logical grouping (eg, seq2seq, seq). They compose modules and may compose sub-models (eg, GAN, learnt meta-optimizer).

**Constructor convention:** Each model has a companion `@dataclass` config (eg, `TransformerConfig`). The constructor takes `config: ConfigClass` and optionally a `tokenizer: Tokenizer`. Models call `self.save_hyperparameters()` for checkpoint serialization.

**Current models:**
- `Transformer` — Seq2seq encoder-decoder for translation. Separate src/tgt vocabularies.
- `HFEncoderDecoder` — Wraps HuggingFace `EncoderDecoderModel` with BERT configs. No tokenizer dependency.
- `Whisper` — Speech-to-text with CNN-based `AudioEncoder` + transformer encoder-decoder. Single vocabulary.

### Datasets (`datasets/`)
Subclasses of `torch.utils.data.Dataset` via `BaseDataset`.

**Constructor convention:** Each dataset has a companion `@dataclass` config (eg, `TranslationDatasetConfig`). The constructor takes `config: ConfigClass`.

**Collation:** Datasets provide a static `get_collate_function()` method that returns a callable for `DataLoader.collate_fn`. Collate functions pad sequences and create attention masks, returning typed batch dataclasses (`TransformerInputBatch`, `WhisperInputBatch`).

**Special token indices** (shared across translation datasets):
```
UNK_IDX = 0, BOS_IDX = 1, EOS_IDX = 2, PAD_IDX = 3
```

**Current datasets:**
- `TextGenDataset` — Sliding-window text generation (PAD_IDX=0 only, no BOS/EOS).
- `TranslationDataset` — Machine translation with SentencePiece tokenization and BPE dropout.
- `TranslationDatasetSpacy` — Translation using Spacy tokenizer + pickle vocab.
- `TranslationDatasetSpacyMulti30K` — Multi30K with torchtext vocab (hardcoded de->en).
- `ATISDataset` — Speech-to-text from mel-spectrograms + SentencePiece transcripts.

### Tokenizers (`tokenizers/`)
Abstract base `Tokenizer` with `tokenize()`/`detokenize()` interface. Currently only `SPTokenizer` (SentencePiece wrapper) is implemented.

- Datasets create their own tokenizer instances internally for training-time tokenization.
- Models receive a `Tokenizer` instance via constructor for inference/evaluation (BLEU reports, interactive translation).

### Metrics (`metrics/`)
Currently contains `bleu.py` with `get_bleu_score()` — computes sentence-level BLEU-4 via NLTK. Used during validation epoch end in Transformer and Whisper models.

Unlike the other registries, the metrics registry collects **callable functions** (not classes). Any top-level function defined in a `.py` file under `metrics/` whose name does not start with `_` is registered.

### Experiment System (`exp/`)
The experiment management system orchestrates training runs:

- **`ExperimentConfig`** — Holds three config dicts (model, dataloaders, trainer) loaded from YAML files. Supports resume from directory/checkpoint.
- **`Experiment`** — Lazily initializes resources (dataloaders, model, trainer) and runs `trainer.fit()`. Uses `multiprocessing.Value`/`Array` for inter-process state communication (state, error buffer, live progress `[epoch, batch, batches_per_epoch, global_step]`). All torch/lightning imports are lazy — the parent process (manager/server) runs on machines without torch. Resolves class names via `registry.models[name]`, `registry.datasets[name]`, `registry.tokenizers[name]`.
- **`ExperimentManager`** — Queue-based manager. Runs experiments sequentially (one at a time on GPU). Supports multi-process mode (child processes) or single-process mode (debugging). Background scheduler polls every 1s to advance the queue. Thread-safe (REST API threads + scheduler share an RLock, never held while waiting on a child). Stop escalates join → SIGTERM → SIGKILL with 5s grace periods. State transitions use compare-and-set under the shared `Value`'s lock so a stop never overwrites a completed/failed terminal state (and vice versa).
- **`exp/callbacks.py`** — Lightning callbacks used inside the child process only: `ExperimentStopper` (graceful stop via shared state) and `ProgressReporter` (publishes live progress to the shared array).
- **`exp/server.py`** — Flask REST API + web UI (`create_app`/`serve`/`python -m exp.server`). See Web UI below.
- **`exp/tb_events.py`** — Dependency-free TensorBoard event-file reader/writer (TFRecord framing + minimal protobuf). The server uses it to serve live metric curves without the `tensorboard` package; `exp/demo.py` uses the writer to fake runs.
- **`exp/machine.py`** — Best-effort machine state (CPU/RAM/disk via psutil, GPUs via pynvml → nvidia-smi fallback); degrades gracefully on machines without NVIDIA tooling.
- **`exp/demo.py`** — UI dev/demo mode: real manager + server, fake child processes (sleep through epochs, stream metrics, honour stops, scripted failures). `uv run python -m exp.demo` — no torch, GPU, or data needed.

#### Web UI (`exp/webui/` + `exp/server.py`)

Self-contained vanilla HTML/CSS/JS dashboard (no build step, no CDN) served by Flask. Polls `/api/status` every 2s and live metrics every 5s. Features: machine tiles (CPU/RAM/disk/GPU meters), current experiment card (state chip, elapsed, epoch/batch progress bar, live loss/BLEU charts with crosshair tooltips), queue reordering, stop/requeue/retry/remove for all buckets, clone-into-new-experiment, create form with YAML editors + `configs/` templates, details modal (metrics, checkpoints, configs), untracked disk-folder panel (resume from `last.ckpt` / delete). Dark and light theme via `prefers-color-scheme`.

Ways to run:
1. Embedded in a sweep script (replaces the trailing `input()`): `from exp.server import serve; serve(exp_manager)` — blocks until Ctrl-C; pass `block=False` for a background thread. Use `host='0.0.0.0'` to reach the UI from another machine.
2. Standalone: `uv run python -m exp.server --master-dir experiments/my-sweep` (enqueue through the UI).
3. Demo: `uv run python -m exp.demo` (fake experiments, for UI development).

REST endpoints all return `{success, ...}` JSON. Mutations are POST with `{index, name}` — the `name` is a stale-state guard; the server refuses if the bucket shifted. Requires the `server` extra: `uv sync --extra server` (flask, psutil).

**IMPORTANT (spawn platforms — macOS/Windows):** any script that enqueues experiments must wrap its top-level code in `if __name__ == '__main__':` — child processes re-execute the main script at bootstrap under spawn. The sweep scripts follow this pattern.

### Configuration (`configs/`)
Each experiment config set is a directory with three YAML files:
- `model.yaml` — Model class name, constructor args, tokenizer config.
- `dls.yaml` — Train/valid dataset class names, constructor args, DataLoader args.
- `trainer.yaml` — PyTorch Lightning Trainer args (epochs, gradient clipping, logging).

Example structure:
```yaml
# model.yaml
class: Transformer
init_args:
  n_blocks: 6
  n_heads: 8
  emb_dim: 512
  attention_type: Vanilla   # prefix used to resolve {prefix}SelfAttention from registry
  ...
tokenizer:
  class: SPTokenizer
  init_args:
    sp_model_path: data/multi30k/m_en_de.model
```

### Sweep Scripts (`test_*.py`)
These are NOT unit tests. They programmatically create `ExperimentConfig` objects with varied hyperparameters and enqueue them into the `ExperimentManager` for sequential execution. Examples:
- `test_multi30k_v1.py` — Sweeps attention types x block counts x vocab sizes (24 experiments).
- `test_multi30k_v1_stability.py` — Repeats a single config 8 times for variance analysis.
- `test_multi30k_v1_misc.py` — Sweeps blocks x heads x embedding dims x weight tying (108 experiments).
- `test_atis_v1.py` — Sweeps blocks x CNN layers x vocab sizes for Whisper (45 experiments).

### Inference Scripts
- `translate.py` — Interactive greedy decoding CLI for Transformer/Whisper.
- `translate_beam.py` — Beam search with curses TUI displaying hypotheses.
- `whisper_preprocess.py` — Audio preprocessing pipeline (mel extraction, normalization).

## Known Issues

These are known bugs/inconsistencies to be addressed in the overhaul:

### Critical
1. **Inconsistent attention return types** — Base-inherited attention modules (vanilla, multi_query, roformer_attn, xformer) return tuples, but standalone modules (stock, average, softmax1) return bare tensors. Transformer blocks index with `[0]`, so standalone modules crash at runtime.
2. **`PositionalEmbedding.forward()` bug** — `return self.encoding[:x]` treats `x` as a scalar, should index by sequence length.
3. **`HFEncoderDecoder` missing `train_losses` init** — `training_step` accesses `self.train_losses` before `on_train_epoch_start` initializes it.

### High
4. **`stock.py` mask not applied** — Creates mask variable but never passes it to `F.scaled_dot_product_attention()`.
5. **`flash.py`/`meme.py` dead code** — Override `get_attention_args()` but stock's `forward()` never calls it.
6. **`xformer.py` returns None for attention weights** — Crashes if `output_attention=True`.
7. **`Whisper` missing `save_hyperparameters()`** — Checkpoints won't properly serialize hparams (present in Transformer and HFEncoderDecoder).
8. **`roformer_attn.py` cross-attention dropout order** — Applies dropout to weights before matmul, inconsistent with self-attention.

### Medium
9. **`TranslationDatasetSpacy` truncation bug** — Truncates raw text string before tokenizing instead of truncating token sequence.
10. **`TranslationDatasetSpacy` missing BOS/EOS on encoder input** — Inconsistent with other translation datasets.
11. **`TranslationDatasetSpacyMulti30K` ignores config language pair** — Always uses hardcoded de->en.
12. **`TranslationDatasetSpacyMulti30K` hardcoded `__len__`** — Returns 29000/1014 regardless of actual dataset size, no 'test' split.
13. **Hardcoded special token indices** — UNK=0, BOS=1, EOS=2, PAD=3 are scattered as module-level constants across files rather than centralized.
