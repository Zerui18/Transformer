from torch.utils.data import Dataset

class BaseDataset(Dataset):
    ''' Base class for all datasets. '''

    @staticmethod
    def get_collate_function() -> callable or None:
        ''' Subclasses should override this method if they want to use a custom collate function for the dataloader. '''
        return None