''' Component registries for zlab.

Builds auto-scanning registries for each subpackage (modules, models, datasets,
metrics, tokenizers) at import time. Each registry is a dict[str, type] mapping
class names to their classes.
'''

import pkgutil
import importlib
import inspect
from pathlib import Path

from torch import nn

from components.models.base_model import BaseModel
from components.datasets.base_dataset import BaseDataset
from components.metrics.base_metric import BaseMetric
from components.tokenizers.base_tokenizer import BaseTokenizer


def _build_registry(package_path: str, package_name: str, base_class: type) -> dict[str, type]:
	''' Scan a package recursively and collect all concrete subclasses of base_class.

	Args:
		package_path: ``str``: filesystem path to the package directory.
		package_name: ``str``: dotted Python package name.
		base_class: ``type``: the abstract base class to match against.
	Returns:
		``dict[str, type]``: class name -> class mapping (excludes abstract classes).
	'''
	registry: dict[str, type] = {}
	for importer, modname, ispkg in pkgutil.walk_packages(
		path=[package_path], prefix=package_name + '.'):
		try:
			module = importlib.import_module(modname)
		except ImportError:
			# skip modules with missing optional dependencies (eg xformers)
			continue
		for name, obj in inspect.getmembers(module, inspect.isclass):
			if (issubclass(obj, base_class)
				and obj is not base_class
				and not inspect.isabstract(obj)):
				registry[name] = obj
	return registry


def _build_modules_registry(package_path: str, package_name: str) -> dict[str, type]:
	''' Scan the modules package for all concrete nn.Module subclasses.

	Excludes abstract base classes (MultiHeadSelfAttentionBase, MultiHeadCrossAttentionBase)
	and classes from other registries (BaseModel subclasses are in the models registry).

	Args:
		package_path: ``str``: filesystem path to the modules package.
		package_name: ``str``: dotted Python package name.
	Returns:
		``dict[str, type]``: class name -> class mapping.
	'''
	registry: dict[str, type] = {}
	for importer, modname, ispkg in pkgutil.walk_packages(
		path=[package_path], prefix=package_name + '.'):
		try:
			module = importlib.import_module(modname)
		except ImportError:
			continue
		for name, obj in inspect.getmembers(module, inspect.isclass):
			if (issubclass(obj, nn.Module)
				and not inspect.isabstract(obj)
				and not issubclass(obj, BaseModel)  # models have their own registry
				and obj.__module__ == module.__name__  # only classes defined in this module
				and obj.__name__ == name):  # skip aliases (e.g. SELF_ATTENTION_CLS)
				registry[name] = obj
	return registry


# Build all registries at import time
_components_dir = Path(__file__).parent

modules_registry: dict[str, type] = _build_modules_registry(
	str(_components_dir / 'modules'), 'components.modules')

models_registry: dict[str, type] = _build_registry(
	str(_components_dir / 'models'), 'components.models', BaseModel)

datasets_registry: dict[str, type] = _build_registry(
	str(_components_dir / 'datasets'), 'components.datasets', BaseDataset)

metrics_registry: dict[str, type] = _build_registry(
	str(_components_dir / 'metrics'), 'components.metrics', BaseMetric)

tokenizers_registry: dict[str, type] = _build_registry(
	str(_components_dir / 'tokenizers'), 'components.tokenizers', BaseTokenizer)
