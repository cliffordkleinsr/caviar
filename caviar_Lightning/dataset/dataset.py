import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from torch import Tensor, is_tensor, tensor
from PIL import Image
from os.path import join
from pandas import DataFrame
import numpy as np
from collections.abc import Callable
from collections import OrderedDict
from ..utilities.utils import Preprocess, def_emb_sz

class ImageDataset(Dataset):
    def __init__(
            self,
            dataframe:DataFrame,
            img_dir:str,
            transforms = None,
            suffix :str = 'jpg'
    ) -> Tensor:
        super().__init__()
        
        self.data = dataframe
        self.transforms = transforms
        self.suffix = suffix
        self.image = self.data.iloc[:, 0]
        self.label = self.data.iloc[:, 7]
        self.img_dir = img_dir
   
    def __len__(self):
        return self.data.shape[0] 
    
    def __getitem__(self, index):
        #image I/O
        single_image = join(self.img_dir, self.image[index])
        img = Image.open(f'{single_image}.{self.suffix}')
        img = self.transforms(img) if self.transforms else img# transform

        label = self.label[index]

        return img if is_tensor(img) else ToTensor()(img), label

class TabularDataset(Dataset):
    def __init__(
            self,
            data: DataFrame = None,
            categorical_cols: list[str] = None,
            continuous_cols: list[str] = None,
            target: str = None,
            processing_strategy: list[Callable] = None 
    ) -> Tensor:
        super().__init__()

        self.data = data
        self.target = target
        if target:
            self.y = data[target].values

        self.processing_strategy = processing_strategy
        self.categorical_cols = [] if categorical_cols is None else categorical_cols
        self.continuous_cols = [] if continuous_cols is None else continuous_cols
        if processing_strategy: 
            self._preprocess_data()
        if len(self.continuous_cols)  != 0:
            self.classes = OrderedDict({n:np.concatenate([['#na#'],c.cat.categories.values])
                                      for n,c in self.data[self.categorical_cols].items()})
        else: self.classes = None

        self.codes_stack = np.stack([c.cat.codes.values for n, c in self.data[self.categorical_cols].items()], 1).astype(np.int64) + 1
        self.conts_stack = np.stack([c.astype('float32').values for n, c in self.data[self.continuous_cols].items()], 1)
        
    def _preprocess_data(self):
       self.data = Preprocess(self.data, self.categorical_cols, self.continuous_cols, self.processing_strategy)

    def get_emb_szs(self, sz_dict=None):
        "Return the default embedding sizes suitable for this data or takes the ones in `sz_dict`."
        return [def_emb_sz(self.classes, n, sz_dict) for n in self.categorical_cols]
     
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        """Generates one sample of data."""
        """Remove` O(n*k)` time complexity `-> Moved to constructor`"""
        # codes = np.stack([c.cat.codes.values for n,c in self.data[self.categorical_cols].items()], 1).astype(np.int64) + 1
        # conts = np.stack([c.astype('float32').values for n,c in self.data[self.continuous_cols].items()], 1)

        cat = tensor(self.codes_stack[index])
        cont = tensor(self.conts_stack[index])
        line = [cat, cont]

        return line, self.y[index]
        """No Dict Return needed"""
            #  return {
            #      "target": self.y[index],
            #      "continuous": (self.categorical_X[index] if self.categorical_cols else Tensor()),
            #      "categorical": (self.categorical_X[index] if self.categorical_cols else Tensor()),
            #  }

            
class ImageTabDataset(Dataset):
    def __init__(self, image_dataset, tabular_dataset):
        """
        Hybrid dataset that integrates image and tabular data.
        """
        self.image_dataset , self.tabular_dataset = image_dataset , tabular_dataset

    def __len__(self):
        return len(self.image_dataset)

    def __getitem__(self, index):
        # Get image and tabular data for the given index
        image_data = self.image_dataset[index]
        tabular_data = self.tabular_dataset[index]

        data = [image_data[0], tabular_data[0]], image_data[1]
        
        return data

    