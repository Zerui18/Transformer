# zlab

A from-scratch, highly customizable PyTorch playground for transformer-like models on DL tasks (and RL in future). Adhere closely to this document in your implementation.

## Stack
- Language: Python 3.12
- Framework: PyTorch + PyTorch Lightning + Einops
- Package manager: uv
- Test framework: None

## Conventions
1. Always clearly define argument and return types in function signatures.
2. Always clearly state the tensor/md-array shapes in both declarations and intermediate computations.
3. Every function/class should have a clear triple-quoted docstring immediately following its declaration. It should follow this Google-style format optimized for IDE tooltip rendering:
```python
''' What the function/class does in one sentence.

[for functions]
Args:
	{name}: ``{type}``: Description.
	{tensor_name}: ``Tensor[(B, T, D), float32]``: Description.
	...

Yields:
	``{type}``: Description.

Returns:
	``{type}``: Description.

[for classes]
Attributes:
	{name}: ``{type}``: Description.
	...

[Additional notes if needed]
'''
```
Types are wrapped in double backticks with a colon separator: `name: ``type``: description`. Tensor types include shape and dtype: ``Tensor[(B, T, D), float32]``.
4. Tensor dtypes are specified as type[+width] (eg, int32/bf16/bool).
5. Tensor shapes are specified as a tuple of symbols (for variable dimensions) and integers (for fixed dimensions), eg, (1, T, C) for fixed batch size of 1, followed by timesteps and channels. In docstrings, tensor shape and dtype are written together as `Tensor[(shape), dtype]`. Use the following capitalised symbols where applicable:
	B: batch size
	T: timesteps
	C: channels
	D: embedding dims
	H: heads
	E: experts
Each symbol can be further specialised with a lowercased subscript, such as Tq for timesteps of query, Tk for timesteps for keys.
6. Prefer using `einops` for complicated tensor reductions and axes permutations as far as possible and follow the aforementioned conventions to make tensor shapes maximally clear.
7. Intersperse function bodies with comments at critical points to aid understanding, but no need to explain every line.
8. Imports should always be in sequence starting from built-in packages, external packages, project imports.
9. Parameters should always be required unless they have a commonly used value or can be None.
10. Functions should start by pre-conditioning tensor shape/dtype constraints where applicable, raising helpful error messages upon failure.
11. Pytorch code should be device agnostic as far as possible. Initializing new tensors should adaptively adopt the dtype and device within their context of operation.
12. For all modules except for `main`, there shall be no top-level statements besides imports and definitions.

## Architecture
The core architectural goal of this project is to empower simple yet rich compositions while remaining fully customizable.

### Project Structure
```
├── components/
│   ├── __init__.py      # Builds registries for each subpackage below
│   ├── modules/		 # Composable building blocks, `torch.nn.Module` subclasses
│   ├── models/		     # End-to-end trainable systems, `BaseModel` subclasses
│   │   └── base_model.py
│   ├── datasets/		 # Data loading and processing, `BaseDataset` subclasses
│   │   └── base_dataset.py
│   ├── metrics/		 # Evaluation metrics, `BaseMetric` subclasses
│   │   └── base_metric.py
│   └── tokenizers/		 # Tokenizers, `BaseTokenizer` subclasses
│       └── base_tokenizer.py
```
Each subpackage may have further subdirectories for logical grouping (eg, attention modules are under `modules/attention/`, seq2seq models are under `models/seq2seq/`, etc).

### Modules (`modules/`)
Subclasses of `torch.nn.Module` and defined under `modules/`, representing reusable and composable building blocks.
Each module's constructor should expose all its configurable hparams. Hparams for nested modules should be exposed in a nested dict with keys corresponding to the argument names of the nested module's constructor.
This allows for rich and deep compositions while maintaining a clear and navigable configuration structure.
Example:
```python
def __init__(self,
             hparam1: type1,
             hparam2: type2 = default2,
             hparam3: type3 | None = None,
             nested_hparams4: dict[str, Any] = {}):
```
Nested hparams may also specify a class name to initialize the nested module.
This should be specified with key `cls` and will be looked-up from the modules registry. 
All other parameters in the dict are passed to the constructor of the looked-up class.
The modules registry is built when the entire package is initialized: It scans the `modules/` directory recursively and collates all torch modules' classes into a dict[str, Class] with keys being the class names stripped of the namespace.

### Models (`models/`)
These are subclasses of the abstract `BaseModel` and defined under `models/`. Compared to modules, each model is a full end-to-end trainable system. The constructor convention is the same as modules, with nested hparams for all nested modules.

Each model must implement two abstract methods from `BaseModel`:
1. `produce(self, batch: dict[str, Any], requested: set[str]) -> dict[str, Any]` — produce requested outputs from a batch. Must always support `'loss'`. May support expensive keys like `'decoded_greedy'` for epoch-level metrics.
2. `supports(self) -> set[str]` — return all possible output keys from `produce()`.

`BaseModel` handles the training loop, optimizer configuration, and metric orchestration. See `base_model.py` for the full contract.

### Datasets (`datasets/`)
These are subclasses of `BaseDataset` and defined under `datasets/`. Each should expose a `get_collate_function()` static method that returns a collate function to be used with the dataset.

### Metrics (`metrics/`)
These are subclasses of `BaseMetric` (which extends `torchmetrics.Metric`) and defined under `metrics/`. Each should implement `_update()` and `compute()`. The public `update()` in `BaseMetric` handles subsampling and should not be overwritten.

Each metric declares:
- `requires: set[str]` — keys needed from `produce()`.
- `frequency: str` — `'step'` (per-batch, default) or `'epoch'` (once at epoch end on sampled data).
- `name: str` — display name for logging (defaults to class name).

See `base_metric.py` for the full interface.

### Tokenizers (`tokenizers/`)
These are subclasses of `BaseTokenizer` and defined under `tokenizers/`. Each should implement `_tokenize()` and `_detokenize()` methods. The public methods `tokenize()` and `detokenize()` are defined in `BaseTokenizer` and should not be overwritten. They handle all the necessary pre- and post-processing steps, such as handling special tokens, padding, truncation, etc. The `_tokenize()` and `_detokenize()` methods should focus solely on the core tokenization logic.

## Experiments
Experiments make use of the `exp` root-level package. Each experiment is defined as a triplet of config files organized under `experiments/{experiment_name}/config/`:
1. `model.yaml` — model class, hparams, optimizer, metrics, checkpoints, and tokenizer.
2. `dataset.yaml` — dataset class and hparams per split, plus dataloader args.
3. `training.yaml` — Lightning Trainer kwargs.

All config files use a flat format with `cls` key for class lookup from registries. Nested modules (optimizer, tokenizer) follow the same `cls` + kwargs pattern. See existing configs for examples.

Launch via `exp.ExperimentManager.create_and_append_experiment(name, config)` where `config` is an `exp.ExperimentConfig` loaded from the three YAML files.

## Harness
Configuration for agent orchestration harness is under `.claude/harness.json`.