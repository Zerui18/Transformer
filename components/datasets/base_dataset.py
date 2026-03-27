from typing import Callable

from torch.utils.data import Dataset


class BaseDataset(Dataset):
	''' Abstract base for all zlab datasets.

	Subclasses must implement __len__() and __getitem__() from
	torch.utils.data.Dataset, and should override get_collate_function()
	if they need a custom collate function for the dataloader.

	The collate function should return a dict[str, Tensor] batch.
	'''

	@staticmethod
	def get_collate_function() -> Callable | None:
		''' Return a collate function for the dataloader, or None for the default.

		Returns:
			``Callable | None``: a function that takes a list of samples and returns a dict[str, Tensor] batch.
		'''
		return None
